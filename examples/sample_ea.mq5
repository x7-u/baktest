//+------------------------------------------------------------------+
//| Sample MA Crossover Expert Advisor                                |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>
CTrade trade;

input int FastPeriod = 10;
input int SlowPeriod = 30;
input double LotSize = 0.1;

int handleFast, handleSlow;
double fastBuf[], slowBuf[];

int OnInit() {
   handleFast = iMA(_Symbol, PERIOD_CURRENT, FastPeriod, 0, MODE_SMA, PRICE_CLOSE);
   handleSlow = iMA(_Symbol, PERIOD_CURRENT, SlowPeriod, 0, MODE_SMA, PRICE_CLOSE);
   ArraySetAsSeries(fastBuf, true);
   ArraySetAsSeries(slowBuf, true);
   return(INIT_SUCCEEDED);
}

void OnTick() {
   CopyBuffer(handleFast, 0, 0, 2, fastBuf);
   CopyBuffer(handleSlow, 0, 0, 2, slowBuf);

   // Crossover: fast crosses above slow
   if(fastBuf[1] <= slowBuf[1] && fastBuf[0] > slowBuf[0]) {
      if(PositionSelect(_Symbol)) {
         trade.PositionClose(_Symbol);
      }
      trade.Buy(LotSize, _Symbol, 0, 0, 0, "Long");
   }

   // Crossunder: fast crosses below slow
   if(fastBuf[1] >= slowBuf[1] && fastBuf[0] < slowBuf[0]) {
      if(PositionSelect(_Symbol)) {
         trade.PositionClose(_Symbol);
      }
      trade.Sell(LotSize, _Symbol, 0, 0, 0, "Short");
   }
}

void OnDeinit(const int reason) {
   IndicatorRelease(handleFast);
   IndicatorRelease(handleSlow);
}
