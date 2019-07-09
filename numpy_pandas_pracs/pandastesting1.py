import pandas as pd
import cupy

s=pd.Series([2,4,7, cupy.nan, 99,78])
print(s)
dates = pd.date_range('20190703', periods=8)
print(dates)


