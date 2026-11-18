# Practical No. 1
# Title: Basic Python data structure - dictionaries, data frames, lists & arrays

import pandas as pd
a = [55, 27, 96]
x = pd.Series(a)
print(x)

import pandas as pd
a = [54, 74, 92]
x = pd.Series(a, index=['a', 'b', 'c'])
print(x)

import pandas as pd
a = {'day1': 11, 'day2': 25}
x = pd.Series(a)
print(x)

import pandas as pd
import numpy as np
array = np.random.randint(10, 20, size=(10, 4))
af = pd.DataFrame(array)
print(af)

import pandas as pd
import numpy as np
array = np.random.randint(10, 20, size=(10, 4))
af = pd.DataFrame(array)
array1 = np.zeros((10, 4), dtype=int)
df = pd.DataFrame(array1)
print(df)

list1 = [[11, 22, 35, 45, 55, 68, 78, 86, 94, 10, 11],
         [18, 69, 10, 11, 12, 18, 26, 83, 64, 85, 66]]
df = pd.DataFrame(list1)
print(df)

t1 = (11, 52, 63, 485, 56)
df = pd.DataFrame(t1)
print(df)

# dictionary
v1 = {'A': [64, 75, 86, 94, 50],
      'B': [10, 20, 30, np.nan, 50],
      'C': np.array([11, 25, 35, 44, 55]),
      'D': np.array([100, 200, 300, 400, 500])
      }
df = pd.DataFrame(v1)
print(df)

df.columns

df.axes

df.index

df.dtypes

df.isna()

df.info()

df.select_dtypes(include=int)

df.select_dtypes(exclude=int)

df.describe()

df.iloc[2:4, 1:3]

df.iloc[2:4, :]

df.iloc[:, 1:3]

df.iloc[:, :]

df.loc[2:2, 'A']

df.head(2)

df.tail(2)

df.shape

df['C'].unique()

df['C'].nunique()

df['A'].value_counts()

df['A'] = df['A'].astype(float)

df

df['B'].mean()

df['B'].median()

df['B'].mode()[0]

df['B'].min()

df['B'].max()

df['B'].std()

df['B'].var()

df['B'].quantile(0.25)

df['B'].quantile(0.50)

df['B'].quantile(0.75)


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
0    55
1    27
2    96
dtype: int64
a    54
b    74
c    92
dtype: int64
day1    11
day2    25
dtype: int64
    0   1   2   3
0  14  10  13  14
1  17  19  11  17
2  16  17  14  12
3  13  17  13  10
4  16  19  11  19
5  19  19  11  18
6  13  17  11  11
7  17  15  12  14
8  14  12  13  12
9  15  11  12  15
   0  1  2  3
0  0  0  0  0
1  0  0  0  0
2  0  0  0  0
3  0  0  0  0
4  0  0  0  0
5  0  0  0  0
6  0  0  0  0
7  0  0  0  0
8  0  0  0  0
9  0  0  0  0
   0   1   2   3   4   5   6   7   8   9   10
0  11  22  35  45  55  68  78  86  94  10  11
1  18  69  10  11  12  18  26  83  64  85  66
     0
0   11
1   52
2   63
3  485
4   56
    A     B   C    D
0  64  10.0  11  100
1  75  20.0  25  200
2  86  30.0  35  300
3  94   NaN  44  400
4  50  50.0  55  500
Index(['A', 'B', 'C', 'D'], dtype='str')
[RangeIndex(start=0, stop=5, step=1), Index(['A', 'B', 'C', 'D'], dtype='str')]
RangeIndex(start=0, stop=5, step=1)
A      int64
B    float64
C      int64
D      int64
dtype: object
       A      B      C      D
0  False  False  False  False
1  False  False  False  False
2  False  False  False  False
3  False   True  False  False
4  False  False  False  False
<class 'pandas.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   A       5 non-null      int64  
 1   B       4 non-null      float64
 2   C       5 non-null      int64  
 3   D       5 non-null      int64  
dtypes: float64(1), int64(3)
memory usage: 292.0 bytes
    A   C    D
0  64  11  100
1  75  25  200
2  86  35  300
3  94  44  400
4  50  55  500
      B
0  10.0
1  20.0
2  30.0
3   NaN
4  50.0
               A          B          C           D
count   5.000000   4.000000   5.000000    5.000000
mean   73.800000  27.500000  34.000000  300.000000
std    17.469974  17.078251  16.970563  158.113883
min    50.000000  10.000000  11.000000  100.000000
25%    64.000000  17.500000  25.000000  200.000000
50%    75.000000  25.000000  35.000000  300.000000
75%    86.000000  35.000000  44.000000  400.000000
max    94.000000  50.000000  55.000000  500.000000
      B   C
2  30.0  35
3   NaN  44
    A     B   C    D
2  86  30.0  35  300
3  94   NaN  44  400
      B   C
0  10.0  11
1  20.0  25
2  30.0  35
3   NaN  44
4  50.0  55
    A     B   C    D
0  64  10.0  11  100
1  75  20.0  25  200
2  86  30.0  35  300
3  94   NaN  44  400
4  50  50.0  55  500
2    86
Name: A, dtype: int64
    A     B   C    D
0  64  10.0  11  100
1  75  20.0  25  200
    A     B   C    D
3  94   NaN  44  400
4  50  50.0  55  500
(5, 4)
array([11, 25, 35, 44, 55])
5
A
64    1
75    1
86    1
94    1
50    1
Name: count, dtype: int64
      A     B   C    D
0  64.0  10.0  11  100
1  75.0  20.0  25  200
2  86.0  30.0  35  300
3  94.0   NaN  44  400
4  50.0  50.0  55  500
np.float64(27.5)
np.float64(25.0)
np.float64(10.0)
np.float64(10.0)
np.float64(50.0)
np.float64(17.07825127659933)
np.float64(291.6666666666667)
np.float64(17.5)
np.float64(25.0)
np.float64(35.0)
"""
