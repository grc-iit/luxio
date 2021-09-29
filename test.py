from clever.common import *

df1 = pd.DataFrame([{'a':1}, {'a':2}])
df2 = pd.DataFrame([{'b':1}, {'b':2}])

print(cartesian_product(df1,df2))
