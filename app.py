"""
Unified Flask web application — supports both Pine Script and MQL5 EA backtesting.
"""

import os
import json
import traceback
import math
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_from_directory, Response
from backtester import Backtester


def sanitize_for_json(obj):
    """Recursively replace NA, NaN, Inf, numpy types with JSON-safe values."""
    if hasattr(obj, '_instance') and not bool(obj):  # PineNA / MQL5NA singleton
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    # Handle numpy types
    import numpy as np
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]
    return obj


app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ENGINE_LABELS = {
    'pine': 'Pine Script',
    'mql5': 'MQL5',
}


@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413


def parse_metatrader_csv(filepath):
    df = None
    for sep in ['\t', ',', ';']:
        try:
            df = pd.read_csv(filepath, sep=sep)
            if len(df.columns) >= 4: break
            df = None
        except Exception:
            continue
    if df is None:
        for sep in ['\t', ',', ';']:
            try:
                df = pd.read_csv(filepath, sep=sep, header=None)
                if len(df.columns) >= 4: break
                df = None
            except Exception:
                continue
    if df is None:
        raise ValueError("Could not parse CSV file. Please check the format.")

    df.columns = [str(c).strip().lower() for c in df.columns]
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ('open', '<open>'): col_map[col] = 'open'
        elif cl in ('high', '<high>'): col_map[col] = 'high'
        elif cl in ('low', '<low>'): col_map[col] = 'low'
        elif cl in ('close', '<close>'): col_map[col] = 'close'
        elif cl in ('volume', 'vol', '<vol>', '<volume>', 'tickvol', '<tickvol>', 'tick_volume'):
            if 'volume' not in col_map.values(): col_map[col] = 'volume'
        elif cl in ('date', '<date>', '<dtyyyymmdd>'): col_map[col] = 'date'
        elif cl in ('time', '<time>', '<time_hh:mm>'): col_map[col] = 'time'
        elif cl in ('datetime', 'date_time'): col_map[col] = 'datetime'
    df = df.rename(columns=col_map)

    if 'open' not in df.columns:
        ncols = len(df.columns)
        if ncols == 6: df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        elif ncols == 7: df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        elif ncols == 8: df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'volume']
        elif ncols == 9: df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread']
        elif ncols == 5: df.columns = ['open', 'high', 'low', 'close', 'volume']
        elif ncols == 4: df.columns = ['open', 'high', 'low', 'close']
        else: raise ValueError(f"Unrecognized CSV format with {ncols} columns.")

    if 'date' in df.columns and 'time' in df.columns and 'datetime' not in df.columns:
        df['datetime'] = df['date'].astype(str) + ' ' + df['time'].astype(str)
    elif 'date' in df.columns and 'datetime' not in df.columns:
        df['datetime'] = df['date'].astype(str)

    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'volume' not in df.columns:
        df['volume'] = 0
    else:
        if isinstance(df['volume'], pd.DataFrame): df['volume'] = df['volume'].iloc[:, 0]
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    return df


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


def _setup_backtest():
    """Parse request data and create a Backtester instance. Returns (bt, engine, label) or raises."""
    if 'csv_file' not in request.files:
        raise ValueError('No CSV file uploaded')
    csv_file = request.files['csv_file']
    if csv_file.filename == '':
        raise ValueError('No file selected')

    engine = request.form.get('engine', 'pine')
    label = ENGINE_LABELS.get(engine, 'Pine Script')
    script = request.form.get('script', '')
    ext = '.mq5' if engine == 'mql5' else '.pine'
    with open(os.path.join(UPLOAD_FOLDER, f'last_script{ext}'), 'w', encoding='utf-8') as f:
        f.write(script)
    if not script.strip():
        raise ValueError(f'No {label} script provided')

    initial_capital = float(request.form.get('initial_capital', 10000))
    commission = float(request.form.get('commission', 0.0))
    commission_per_lot = float(request.form.get('commission_per_lot', 0.0))
    commission_per_trade = float(request.form.get('commission_per_trade', 0.0))
    default_qty = float(request.form.get('default_qty', 1.0))
    risk_pct = float(request.form.get('risk_pct', 0.0))
    spread_pips = float(request.form.get('spread_pips', 0.0))
    slippage_pips = float(request.form.get('slippage_pips', 0.0))
    max_bars = int(request.form.get('max_bars', 0))
    base_tf = request.form.get('base_tf', 'M5')
    utc_offset = int(request.form.get('utc_offset', 0))

    # Validate inputs
    if initial_capital <= 0:
        raise ValueError('Starting balance must be positive')
    if commission < 0 or commission_per_lot < 0 or commission_per_trade < 0:
        raise ValueError('Commission values cannot be negative')
    if spread_pips < 0:
        raise ValueError('Spread cannot be negative')
    if slippage_pips < 0:
        raise ValueError('Slippage cannot be negative')
    if default_qty <= 0:
        raise ValueError('Default quantity must be positive')

    filepath = os.path.join(UPLOAD_FOLDER, 'data.csv')
    csv_file.save(filepath)
    df = parse_metatrader_csv(filepath)

    if max_bars > 0 and len(df) > max_bars:
        df = df.tail(max_bars).reset_index(drop=True)

    # Date exclusion filters (applied post-backtest to remove trades, not data)
    date_filters = json.loads(request.form.get('date_filters', '[]'))

    if len(df) < 2:
        raise ValueError('CSV file has insufficient data (need at least 2 bars)')

    df_smt = None
    if 'smt_file' in request.files:
        smt_file = request.files['smt_file']
        if smt_file.filename:
            smt_path = os.path.join(UPLOAD_FOLDER, 'smt_data.csv')
            smt_file.save(smt_path)
            df_smt = parse_metatrader_csv(smt_path)
            if max_bars > 0 and len(df_smt) > max_bars:
                df_smt = df_smt.tail(max_bars).reset_index(drop=True)
            min_len = min(len(df), len(df_smt))
            df = df.tail(min_len).reset_index(drop=True)
            df_smt = df_smt.tail(min_len).reset_index(drop=True)

    # Additional correlation symbols
    extra_dfs = []
    extra_files = request.files.getlist('extra_files')
    for idx, ef in enumerate(extra_files):
        if ef.filename:
            ep = os.path.join(UPLOAD_FOLDER, f'extra_{idx}.csv')
            ef.save(ep)
            edf = parse_metatrader_csv(ep)
            if max_bars > 0 and len(edf) > max_bars:
                edf = edf.tail(max_bars).reset_index(drop=True)
            extra_dfs.append(edf)

    funded_rules = None
    if request.form.get('funded_enabled') == '1':
        funded_rules = {
            'enabled': True,
            'target': float(request.form.get('funded_target', 10)),
            'max_dd': float(request.form.get('funded_max_dd', 10)),
            'daily_dd': float(request.form.get('funded_daily_dd', 5)),
            'min_days': int(request.form.get('funded_min_days', 5)),
        }

    bt = Backtester(
        data=df, source=script, engine=engine,
        initial_capital=initial_capital,
        commission_pct=commission,
        commission_per_lot=commission_per_lot,
        commission_per_trade=commission_per_trade,
        default_qty=default_qty,
        risk_pct=risk_pct,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        smt_data=df_smt,
        extra_data=extra_dfs,
        base_tf=base_tf,
        utc_offset=utc_offset,
        date_filters=date_filters,
        funded_rules=funded_rules,
    )
    return bt, engine, label


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    try:
        bt, engine, label = _setup_backtest()
        results = bt.run()

        interp_name = type(bt.interpreter).__name__
        results['engine'] = 'cython' if 'Fast' in interp_name else 'python'
        results['engine_type'] = engine

        return jsonify(sanitize_for_json(results))

    except SyntaxError as e:
        return jsonify({'error': f'Syntax error: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Data error: {str(e)}'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Backtest error: {str(e)}'}), 500


# ─── MetaTrader 5 Routes ──────────────────────────────────────────────────────

@app.route('/api/mt5/status')
def mt5_status():
    """Check if MT5 terminal is available."""
    try:
        from mt5_source import MT5DataSource
        src = MT5DataSource()
        ok = src.initialize()
        if ok:
            src.shutdown()
            return jsonify({'connected': True})
        return jsonify({'connected': False, 'error': 'MT5 terminal not running'})
    except ImportError:
        return jsonify({'connected': False, 'error': 'MetaTrader5 package not installed'})
    except Exception as e:
        return jsonify({'connected': False, 'error': str(e)})


@app.route('/api/mt5/symbols')
def mt5_symbols():
    """Get available symbols from MT5 broker."""
    try:
        from mt5_source import MT5DataSource
        src = MT5DataSource()
        if not src.initialize():
            return jsonify({'symbols': [], 'error': 'MT5 terminal not running'})
        try:
            filter_text = request.args.get('filter', '')
            symbols = src.list_symbols(filter_text)
            return jsonify({'symbols': symbols})
        finally:
            src.shutdown()
    except ImportError:
        return jsonify({'symbols': [], 'error': 'MetaTrader5 package not installed'})
    except Exception as e:
        return jsonify({'symbols': [], 'error': str(e)})


@app.route('/api/mt5/fetch', methods=['POST'])
def mt5_fetch_and_backtest():
    """Fetch data from MT5 and run backtest."""
    try:
        from mt5_source import MT5DataSource
        from datetime import datetime as _dt

        engine = request.form.get('engine', 'pine')
        label = ENGINE_LABELS.get(engine, 'Pine Script')
        script = request.form.get('script', '')
        if not script.strip():
            return jsonify({'error': f'No {label} script provided'}), 400

        symbol = request.form.get('symbol', '').strip()
        if not symbol:
            return jsonify({'error': 'No symbol specified'}), 400

        tf_string = request.form.get('base_tf', 'M5')
        date_from_str = request.form.get('date_from', '')
        date_to_str = request.form.get('date_to', '')

        try:
            date_from = _dt.strptime(date_from_str, '%Y-%m-%d')
            date_to = _dt.strptime(date_to_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

        # Parse settings
        initial_capital = float(request.form.get('initial_capital', 10000))
        commission = float(request.form.get('commission', 0.0))
        commission_per_lot = float(request.form.get('commission_per_lot', 0.0))
        commission_per_trade = float(request.form.get('commission_per_trade', 0.0))
        default_qty = float(request.form.get('default_qty', 1.0))
        risk_pct = float(request.form.get('risk_pct', 0.0))
        spread_pips = float(request.form.get('spread_pips', 0.0))
        slippage_pips = float(request.form.get('slippage_pips', 0.0))
        utc_offset = int(request.form.get('utc_offset', 0))
        max_bars = int(request.form.get('max_bars', 0))
        date_filters = json.loads(request.form.get('date_filters', '[]'))

        # Validate
        if initial_capital <= 0:
            return jsonify({'error': 'Starting balance must be positive'}), 400
        if default_qty <= 0:
            return jsonify({'error': 'Default quantity must be positive'}), 400

        # Fetch from MT5
        src = MT5DataSource()
        if not src.initialize():
            return jsonify({'error': 'MetaTrader 5 terminal is not running. Please start MT5 and log in.'}), 503
        try:
            df = src.fetch_bars(symbol, tf_string, date_from, date_to)
        except (ValueError, RuntimeError) as e:
            return jsonify({'error': str(e)}), 400
        finally:
            src.shutdown()

        if max_bars > 0 and len(df) > max_bars:
            df = df.tail(max_bars).reset_index(drop=True)
        if len(df) < 2:
            return jsonify({'error': 'Insufficient data fetched (need at least 2 bars)'}), 400

        # Optional SMT symbol fetch
        df_smt = None
        smt_symbol = request.form.get('smt_symbol', '').strip()
        if smt_symbol:
            src2 = MT5DataSource()
            if src2.initialize():
                try:
                    df_smt = src2.fetch_bars(smt_symbol, tf_string, date_from, date_to)
                    if max_bars > 0 and len(df_smt) > max_bars:
                        df_smt = df_smt.tail(max_bars).reset_index(drop=True)
                    min_len = min(len(df), len(df_smt))
                    df = df.tail(min_len).reset_index(drop=True)
                    df_smt = df_smt.tail(min_len).reset_index(drop=True)
                except Exception:
                    df_smt = None
                finally:
                    src2.shutdown()

        # Funded account rules
        funded_rules = None
        if request.form.get('funded_enabled') == '1':
            funded_rules = {
                'enabled': True,
                'target': float(request.form.get('funded_target', 10)),
                'max_dd': float(request.form.get('funded_max_dd', 10)),
                'daily_dd': float(request.form.get('funded_daily_dd', 5)),
                'min_days': int(request.form.get('funded_min_days', 5)),
            }

        # Run backtest (same path as CSV)
        bt = Backtester(
            data=df, source=script, engine=engine,
            initial_capital=initial_capital,
            commission_pct=commission,
            commission_per_lot=commission_per_lot,
            commission_per_trade=commission_per_trade,
            default_qty=default_qty,
            risk_pct=risk_pct,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            smt_data=df_smt,
            base_tf=tf_string,
            utc_offset=utc_offset,
            date_filters=date_filters,
            symbol_name=symbol,
            funded_rules=funded_rules,
        )
        results = bt.run()

        interp_name = type(bt.interpreter).__name__
        results['engine'] = 'cython' if 'Fast' in interp_name else 'python'
        results['engine_type'] = engine
        results['data_source'] = f'MT5: {symbol} ({tf_string})'

        return jsonify(sanitize_for_json(results))

    except ImportError:
        return jsonify({'error': 'MetaTrader5 package not installed. Run: pip install MetaTrader5'}), 500
    except SyntaxError as e:
        return jsonify({'error': f'Syntax error: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Data error: {str(e)}'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Backtest error: {str(e)}'}), 500


@app.route('/api/backtest-stream', methods=['POST'])
def run_backtest_stream():
    try:
        bt, engine, label = _setup_backtest()

        def generate():
            for msg in bt.run_streaming():
                if msg['status'] == 'done':
                    results = msg['results']
                    interp_name = type(bt.interpreter).__name__
                    results['engine'] = 'cython' if 'Fast' in interp_name else 'python'
                    results['engine_type'] = engine
                    yield f"data: {json.dumps(sanitize_for_json(msg))}\n\n"
                else:
                    yield f"data: {json.dumps(msg)}\n\n"

        return Response(generate(), mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _apply_params(script, params, engine):
    """Apply parameter substitutions to a script."""
    modified = script
    for name, val in params.items():
        if engine == 'mql5':
            modified = re.sub(
                rf'(input\s+\w+\s+{re.escape(name)}\s*=\s*)\d+',
                rf'\g<1>{val}', modified)
        else:
            modified = re.sub(
                rf"({re.escape(name)}\s*=\s*input(?:\.int|\.float)?\s*\()\d+",
                rf'\g<1>{val}', modified)
    return modified


def _run_single_backtest(df, script, engine, initial_capital, commission,
                          commission_per_lot, commission_per_trade, default_qty,
                          risk_pct_val, spread_pips, slippage_pips, smt_data, metric_name):
    """Run a single backtest and return the optimization metric value.
    Designed to be called from ProcessPoolExecutor."""
    try:
        bt = Backtester(data=df, source=script, engine=engine,
                        initial_capital=initial_capital, commission_pct=commission,
                        commission_per_lot=commission_per_lot,
                        commission_per_trade=commission_per_trade,
                        default_qty=default_qty, risk_pct=risk_pct_val, spread_pips=spread_pips,
                        slippage_pips=slippage_pips, smt_data=smt_data)
        r = bt.run()
        return r['metrics'].get(metric_name, 0) or 0
    except Exception:
        return None


@app.route('/api/optimize', methods=['POST'])
def run_optimize():
    try:
        from itertools import product

        # Parse common params
        engine = request.form.get('engine', 'pine')
        script = request.form.get('script', '')
        if not script.strip():
            return jsonify({'error': 'No script provided'}), 400

        initial_capital = float(request.form.get('initial_capital', 10000))
        commission = float(request.form.get('commission', 0.0))
        commission_per_lot = float(request.form.get('commission_per_lot', 0.0))
        commission_per_trade = float(request.form.get('commission_per_trade', 0.0))
        default_qty = float(request.form.get('default_qty', 1.0))
        spread_pips = float(request.form.get('spread_pips', 0.0))
        slippage_pips = float(request.form.get('slippage_pips', 0.0))

        train_bars = int(request.form.get('train_bars', 5000))
        test_bars = int(request.form.get('test_bars', 1000))
        step_bars = int(request.form.get('step_bars', 1000))
        metric_name = request.form.get('metric', 'sharpe_ratio')

        param_ranges = json.loads(request.form.get('param_ranges', '[]'))

        # Parse CSV
        csv_file = request.files.get('csv_file')
        if not csv_file:
            return jsonify({'error': 'No CSV file'}), 400
        filepath = os.path.join(UPLOAD_FOLDER, 'data.csv')
        csv_file.save(filepath)
        df = parse_metatrader_csv(filepath)

        df_smt = None
        if 'smt_file' in request.files:
            smt_file = request.files['smt_file']
            if smt_file.filename:
                smt_path = os.path.join(UPLOAD_FOLDER, 'smt_data.csv')
                smt_file.save(smt_path)
                df_smt = parse_metatrader_csv(smt_path)

        # Generate parameter combinations
        param_names = [p['name'] for p in param_ranges]
        param_values = [list(range(int(p['start']), int(p['end']) + 1, int(p['step']))) for p in param_ranges]
        combos = list(product(*param_values)) if param_values else [()]

        windows = []
        total_oos_pnl = 0
        total_oos_trades = 0
        total_oos_wins = 0
        best_overall = {}

        # Walk-forward loop
        for win_start in range(0, len(df) - train_bars - test_bars + 1, step_bars):
            train_df = df.iloc[win_start:win_start + train_bars].reset_index(drop=True)
            test_df = df.iloc[win_start + train_bars:win_start + train_bars + test_bars].reset_index(drop=True)

            train_smt = None
            test_smt = None
            if df_smt is not None:
                train_smt = df_smt.iloc[win_start:win_start + train_bars].reset_index(drop=True)
                test_smt = df_smt.iloc[win_start + train_bars:win_start + train_bars + test_bars].reset_index(drop=True)

            best_metric = -999999
            best_params = {}

            # Sweep parameters on train data (parallel if >1 combo)
            if len(combos) > 1:
                # Parallel sweep using ProcessPoolExecutor
                futures = {}
                with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, len(combos))) as executor:
                    for combo in combos:
                        params = dict(zip(param_names, combo))
                        mod_script = _apply_params(script, params, engine)
                        future = executor.submit(
                            _run_single_backtest, train_df, mod_script, engine,
                            initial_capital, commission, commission_per_lot,
                            commission_per_trade, default_qty, risk_pct,
                            spread_pips, slippage_pips, train_smt, metric_name)
                        futures[future] = params
                    for future in as_completed(futures):
                        try:
                            m_val = future.result()
                            if m_val is not None and m_val > best_metric:
                                best_metric = m_val
                                best_params = futures[future]
                        except Exception as opt_err:
                            import traceback
                            traceback.print_exc()
                            continue
            else:
                # Single combo — no parallelism needed
                for combo in combos:
                    params = dict(zip(param_names, combo))
                    mod_script = _apply_params(script, params, engine)
                    try:
                        bt_train = Backtester(data=train_df, source=mod_script, engine=engine,
                                              initial_capital=initial_capital, commission_pct=commission,
                                              commission_per_lot=commission_per_lot,
                                              commission_per_trade=commission_per_trade,
                                              default_qty=default_qty, risk_pct=risk_pct, spread_pips=spread_pips,
                                              slippage_pips=slippage_pips, smt_data=train_smt)
                        r = bt_train.run()
                        m_val = r['metrics'].get(metric_name, 0) or 0
                        if m_val > best_metric:
                            best_metric = m_val
                            best_params = params
                    except Exception as opt_err:
                        import traceback
                        traceback.print_exc()
                        continue

            # Test with best params
            modified_script = _apply_params(script, best_params, engine)

            try:
                bt_test = Backtester(data=test_df, source=modified_script, engine=engine,
                                      initial_capital=initial_capital, commission_pct=commission,
                                      commission_per_lot=commission_per_lot,
                                      commission_per_trade=commission_per_trade,
                                      default_qty=default_qty, risk_pct=risk_pct, spread_pips=spread_pips,
                                      slippage_pips=slippage_pips, smt_data=test_smt)
                test_r = bt_test.run()
                test_m = test_r['metrics']

                windows.append({
                    'train_metric': round(best_metric, 2),
                    'best_params': best_params,
                    'test_trades': test_m['total_trades'],
                    'test_pnl': test_m['net_profit'],
                    'test_wr': test_m['win_rate'],
                })
                total_oos_pnl += test_m['net_profit']
                total_oos_trades += test_m['total_trades']
                total_oos_wins += test_m['winning_trades']
            except Exception as opt_err:
                import traceback
                traceback.print_exc()
                windows.append({
                    'train_metric': round(best_metric, 2),
                    'best_params': best_params,
                    'test_trades': 0, 'test_pnl': 0, 'test_wr': 0,
                })

            if not best_overall:
                best_overall = best_params

        avg_oos_wr = (total_oos_wins / total_oos_trades * 100) if total_oos_trades > 0 else 0

        return jsonify(sanitize_for_json({
            'windows': windows,
            'total_oos_pnl': round(total_oos_pnl, 2),
            'avg_oos_wr': round(avg_oos_wr, 1),
            'best_overall': best_overall,
        }))

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Optimization error: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  Unified Backtester - http://localhost:1234")
    print("  Engines: Pine Script + MQL5")
    print("=" * 60)
    app.run(debug=False, port=1234)
