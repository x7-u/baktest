"""
Unified Flask web application — supports both Pine Script and MQL5 EA backtesting.
"""

import os
import json
import traceback
import math
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, Response
from backtester import Backtester


def sanitize_for_json(obj):
    """Recursively replace NA, NaN, Inf with None for JSON serialization."""
    if hasattr(obj, '_instance') and not bool(obj):  # PineNA / MQL5NA singleton
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
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
    spread_pips = float(request.form.get('spread_pips', 0.0))
    slippage_pips = float(request.form.get('slippage_pips', 0.0))
    max_bars = int(request.form.get('max_bars', 0))
    base_tf = request.form.get('base_tf', 'M5')

    filepath = os.path.join(UPLOAD_FOLDER, 'data.csv')
    csv_file.save(filepath)
    df = parse_metatrader_csv(filepath)

    if max_bars > 0 and len(df) > max_bars:
        df = df.tail(max_bars).reset_index(drop=True)
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

    bt = Backtester(
        data=df, source=script, engine=engine,
        initial_capital=initial_capital,
        commission_pct=commission,
        commission_per_lot=commission_per_lot,
        commission_per_trade=commission_per_trade,
        default_qty=default_qty,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        smt_data=df_smt,
        extra_data=extra_dfs,
        base_tf=base_tf,
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


@app.route('/api/optimize', methods=['POST'])
def run_optimize():
    try:
        import re
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

            # Sweep parameters on train data
            for combo in combos:
                modified_script = script
                params = {}
                for name, val in zip(param_names, combo):
                    params[name] = val
                    if engine == 'mql5':
                        modified_script = re.sub(
                            rf'(input\s+\w+\s+{re.escape(name)}\s*=\s*)\d+',
                            rf'\g<1>{val}', modified_script)
                    else:
                        modified_script = re.sub(
                            rf"(input(?:\.int|\.float)?\s*\([^)]*(?:title\s*=\s*['\"]){re.escape(name)}['\"][^)]*defval\s*=\s*)\d+",
                            rf'\g<1>{val}', modified_script)
                        modified_script = re.sub(
                            rf"({re.escape(name)}\s*=\s*input(?:\.int|\.float)?\s*\()\d+",
                            rf'\g<1>{val}', modified_script)

                try:
                    bt_train = Backtester(data=train_df, source=modified_script, engine=engine,
                                          initial_capital=initial_capital, commission_pct=commission,
                                          commission_per_lot=commission_per_lot,
                                          commission_per_trade=commission_per_trade,
                                          default_qty=default_qty, spread_pips=spread_pips,
                                          slippage_pips=slippage_pips, smt_data=train_smt)
                    r = bt_train.run()
                    m_val = r['metrics'].get(metric_name, 0)
                    if m_val is None:
                        m_val = 0
                    if m_val > best_metric:
                        best_metric = m_val
                        best_params = params
                except Exception:
                    continue

            # Test with best params
            modified_script = script
            for name, val in best_params.items():
                if engine == 'mql5':
                    modified_script = re.sub(
                        rf'(input\s+\w+\s+{re.escape(name)}\s*=\s*)\d+',
                        rf'\g<1>{val}', modified_script)
                else:
                    modified_script = re.sub(
                        rf"({re.escape(name)}\s*=\s*input(?:\.int|\.float)?\s*\()\d+",
                        rf'\g<1>{val}', modified_script)

            try:
                bt_test = Backtester(data=test_df, source=modified_script, engine=engine,
                                      initial_capital=initial_capital, commission_pct=commission,
                                      commission_per_lot=commission_per_lot,
                                      commission_per_trade=commission_per_trade,
                                      default_qty=default_qty, spread_pips=spread_pips,
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
            except Exception:
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
