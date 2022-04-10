---
layout: post
title:  "数据分析 data analysis"
date:   2022-03-030 2:00:00 +0000
categories: jekyll update
---

# 数据分析

> 赠给我亲爱的妻子cocoa，感谢她为家庭的辛苦付出和默默奉献！

当代是一个数据的时代，未来很长一段时间，数据为王的现状依然会持续下去。数字建国、免费经济、东数西算、大数据计算、人工智能这些各个领域耳熟能详的新鲜名词无不例外的和数据的收集分析紧密的联系在一起。数据分析大概分以下几个相互关联的重要阶段：

1. 数据搜集/数据采集

数据的本质是信息，凡是可以用某种方式被观察记录下来的信息都是数据，例如：数据库中包含数据的任意表格、应用程序在屏幕上的打印日志、股票的股价波动记录、互联网商城里商品的成交订单、健康宝记录的行动轨迹、某人遛狗喂鸟的时间规律等等。数据的量可多、可少，可以是阶段性的呈现、也可以是连续不断的数据流形式，但数据的分析一般都是针对一段时间内的分析，得出结论或者进行持续调整的预测。以尽可能模式化的形式记录数据，保证其连续性，保存在特定位置可以查询到的阶段就是数据搜集/采集阶段。

2. 数据清洗

数据清洗阶段的目的是对搜集到的数据进行检查，通过筛选、变换、标准化、新增等手段“清洗掉脏数据”，确保特征数据的表现一致性和有效性，为后面的数据挖掘做准备。数据清洗阶段经常遇到的问题和处理方法如下：

- 各种数据错误：改正数据源，或匹配抽取有效数据，或删除无效数据
- 异常值：过滤出有效数据，或填充默认值，或计算出新数据
- 重复值：根据业务特征不做处理，或做处理：去重、计算出新数据
- 空值等：统计，填充默认值，或计算出新数据
- 数据颗粒度太小：归类、聚合出新数据

生产环境中，可能会遇到从多个、多种数据源获取同一类数据的情况，很难避免数据格式的不统一性，因此，数据清洗阶段除了清洗脏数据外，还会有数据变换、分类、集成、装载等过程，形成最终的数据仓库（数仓）持久化和运维管理起来，在数据仓库的基础上建设数据中台对上层提供服务。

3. 数据挖掘

根据数据的规律性进行提炼、归纳、总结，可以得出阶段性结果或者做出预测性分析，因此可以对风险提前做好准备、及时调整决策、创造出商业价值等。数据挖掘阶段是将数据形态升华到知识并创造价值的过程，通常包含下面几个重要步骤：

- 充分理解产生数据的业务逻辑：对业务逻辑和数据了解是数据分析的基础，对基于业务逻辑产生的数据规律越敏感，数据挖掘的效率和准确性就越高，因此数据分析通常是跨行业的领域，不仅要了解数据分析系统和工具（通常是计算机行业领域），还要熟悉业务逻辑以及数据意义。
- 数据建模：根据“我想这样这样使用数据”的目标，建立一个或多个可反复验证的模型，然后利用一部分数据或独立的验证专用数据集对模型进行测试，根据测试结果调整模型，再用更多的数据进行训练、测试，周而复始，最终挑选或得出最合适的模型。
- 数据评估：建模阶段的数据是有限的，并且一定和实际中的数据有偏差，数据评估阶段着重于在实际应用中使用模型后，对测试的结果进行评估，定义模型的价值。数据评估后的模型可能会回炉重新调整，或推广到更大范围进行使用。

4. 数据展示/数据可视化

数据展示用简单直观的形式，把数据分析的结果展示出来，帮助数据决策者理解数据的特征和规律。数据展示的形式可以是文本、表格和图形，相对于繁多复杂的数据，数据展示的内容更加简洁、直观、容易理解，通常以报告、论文、大屏等形式出现。

数据图形展示的种类最为丰富，经常用到的种类包括：折线、条形图、直方图、饼图、散点图、雷达图等，根据不同的数据展示目的可以选用不同的图形展示。


# 数据分析工具之pandas

pandas是使用python语言编写的最出名的数据分析工具之一，特别擅长于处理表格类型的静态数据（tabular data）。

## 安装pandas和相关工具
在python环境以及安装好的主机上运行下面的命令：

```
pip install pandas numpy matplotlib openpyxl
```
- pandas：数据分析工具
- numpy: 数学函数库，支持维度数组和矩阵计算
- matplotlib: 数据可视化工具
- openpyxl: excel读取工具

## 特征语句

使用pandas的python文件中通常加入下列特征语句：

```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

## 数据输入输出（数据搜集）
pandas提供业内很多标准类型数据的输入输出方法，可以交互csv, excel, json, xml, sql等常用数据存储源。以excel为例，使用方法是：

```
# 读取数据
df1 = pd.read_excel('C:\\Users\\qitia\\Documents\\大宗成交.xlsx', sheet_name='Sheet1')
```
```
# 如果数据源表中存在多个sheet, 标准用法有如下两种：
## 1. 使用ExcelFile类class
xlsx = pd.ExcelFile('path_to_file.xls')
df_sheet1 = pd.read_excel(xlsx, 'Sheet1')

## 2. 同时使用上下文管理器context manager
data = {}
with pd.ExcelFile('path_to_file.xls') as xls:
    data['Sheet1'] = pd.read_excel(xls, 'Sheet1', index_col=None, na_values=['NA'])
    data['Sheet2'] = pd.read_excel(xls, 'Sheet2', index_col=1)
```
```
# 把数据写入excel文件
df.to_excel('C:\\Users\\qitia\\Documents\\大宗成交_分析结果.xlsx', index_label='label', merge_cells=False)

```

## 数据结构

**DataFrame和Series**

![pandas]({{ site.url }}/images/pandas-1.png)

pandas中引入的最重要的两个新的数据类型是`数据帧DataFrame`和`数据序列Series`。DataFrame是二维的表格对象，由表格首行的表头、首列的索引和表格里面的二维数组内容组成，Series是有索引、有数据内容的一维数组对象。在科学计算和数据分析领域，numpy和pandas的联系很紧密，一维和多维`数组Array`的概念是numpy组件介绍的，所谓数组就是一个有序的元素序列，数组的一般特征是存储了相同类型数据的集合（不同类型的异类数据就是脏数据，需要清洗或单独处理）。 

pandas对DataFrame的操作，是对二维数组形式的所有数据内容的全体操作，效率很高，返回的对象大多数情况下是一个新的DataFrame，只有当操作返回的是只有一列的数据时，返回的是一个新的Series，降维操作。DataFrame和Series的方法有不一样的地方，注意区别对待。

## 数据检查
### 获取基本信息
```
# 索引列表
df1.index
# 表头列表
df1.columns 或者 df1.keys()
# 数据矩阵
df1.values
# 以表头为索引内容的数据类型，返回Series
df1.dtypes
# 数据矩阵的数据量
df1.shape
df1.size

# 简报，返回以表头为索引的数据简报，返回DataFrame
df1.info()

# 视检数据
df1.head()
df1.tail()
```

### 数据筛选
```
# 根据表头筛选特定一列或多列数据
df1['交易日期'] 或者 df1.交易日期
df1[['交易日期', '股票代码']]
# 根据索引筛选特定一行或多行数据
df1.iloc[1]
df1.iloc[1:5, '成交量']
df1.loc[40:, ['交易日期', '股票代码','成交量（万股）']]

# 根据数据特征筛选数据区域
## 相当于excel的filter并扩展其它列
df1[df1['成交价格'] >= 50]
df1[df1.股票代码.isin([300308, 300759])]
df1[df1['成交价格'].between(18, 22)]
df1[(df1['交易日期'] == '2022-03-29') | (df1['成交量（万股）'] <= 10)]
df1[df1.溢价率.notna()]

## 筛选并使用部分表头展示数据：需要使用.loc/.iloc
df1.loc[df1['最新价'] >= 100, ['股票代码', '交易日期']]
df1.iloc[1:3, 2:5]
```
## 数据变换
### 数据类型转换
```
# 数字浮点、整形以及和字符串转换，数字可以进行运算，字符串可以进行匹配、模糊搜索

## 使用astype
df1.astype({'最新价': 'float64'})
df1.astype({'成交量（万股）': 'int32'}, errors='raise')
df1['成交量（万股）'].astype('string')

## 使用pd.to_numeric，返回是Series
ser1 = df1['溢价率']
pd.to_numeric[ser1, downcast='float']
pd.to_numeric[ser1, errors='coerce', downcast='unsigned'] #downcast节省内存；errors='corece'不能转换则置空

# 数字的格式转换

## 小数点位数，遇到其它如字符串列则不处理
df1.round(2)
df1.round({'成交价格': 1, '溢价率': 4})

## 小数和百分数转换
df1['溢价率'].map('{:.2%}'.format)
df1['溢价百分比'].map(lambda x: float(x.strip('%')) / 100)
```
### 修改数据（数据挖掘）
```
#  新增列数据
## 完全新的一列
df.insert(2, '日期', pd.date_range(start='2021-12-27', end='2022-1-5', freq='B'), allow_duplicates=False)

## 计算出来的一列/多列
df1['成交额（万元)'] = df1['成交价格'] * df1['成交量（万股）']
df1['股票描述'] =  '大宗交易_' + df1['股票简称']
df2 = df1.assign(价差 = df1['最新价'] - df1['成交价格'], 溢价率百分比 = df1['溢价率'] * 100)

## 修改某列（/某行）对应的数据
df1.at[0:9, '股票描述'] = '需要重点关注'
df1['溢价率'] = df1['溢价率'] * 100

## 删除列数据
df2 = df1[df1.columns.difference(['股票描述'])]
### drop方法默认axis=0，用于删除行；删除列时传参数axis=1
df1.drop('股票描述', axis=1, inplace=True)

# 新增行数据
# 计算出新的一行/多行
df2 = df1.drop('交易日期', axis=1)
df2.loc['新增行序列号名'] = df2.iloc[0] + df2.iloc[1]

df3 = pd.concat([df1, df2], ignore_index=True)

## 删除行数据
df2 = df1.drop(45)

## apply方法
### apply里面的方法可以是同纬度的例如:np.sqrt，也可以是降维的np.sum，返回数据类型可能不一样
### 接受lambda函数对象，但只能传一个参数x
df1.apply(np.sum)
df1[['成交量（万股）', '成交价格']].apply({'成交量（万股）': np.sum,  '成交价格': np.mean})


```

### 缺失值、空值处理（数据清洗）
```
# 检查表中是否存在任意数据缺失
# 注意: ''和' '都不是缺失值而是字符串
df1.isna().values.any()
df1.notna()

# 删除缺失值
## 整行的数据都缺失就删除; dropna也可以用于行和列
df1.dropna(how='all', inplace=True)
## 用默认值填充缺失值，是dropna的反义词
df1.fillna(0)

# 辅助函数any()和all()
## 判断是否所有数据是True（非空值）
df1.all()
df1.all(axis='columns')
## 判断是否任意数据是True
df1.any()
df1.any(axis=None)

# 找出缺失值的位置
for i in df1.columns:
    if df1[i].count() != len(df1):
        row = df1[i][df1[i].isnull().values].index.tolist()
        print('列名：'{}', 第{}行位置有缺失值'.format(i,row))
```

### 重复数据（数据清洗）
```
#  查看数据行是否重复
df1.duplicated()
## 查看特定列里的内容是否重复，默认保留第一个数据，剩下重复的数据都标记为True
df1.duplicated(subset=['股票代码'], keep='first')

# 计算重复数据的个数
df1[df1.duplicated(subset=['股票代码', '成交价格'], keep='first')].count()
## 重复数据
df1[df1.duplicated()]

# 去重
df1[-df1.duplicated()]
df1.drop_duplicates(subset=['股票代码'], keep='last')

## Series去重后的数据列表
df1['交易日期'].unique()
## Series去重后的数据列表长度
df1['交易日期'].nunique()
## Series重复的数据以及对应的重复次数
df1['交易日期'].value_counts()
```
### 排序
```
# 按照特定列内容扩展排序
df1.sort_values(by=['股票代码', '交易日期'], ascending=False, na_position='first')
df1.sort_values(by='股票简称', key=lambda col: col.str.upper())
```
### 时间类型.dt.的数据清洗
```
# 日期序列Series
## 创建间隔特定频率的日期
pd.date_range(start='2021-12-27', end='2022-12-27', freq='Q')
pd.date_range(start='2021-12-27', freq='2W', periods=3)
## 创建工作日
pd.date_range(start='2021-12-27', end='2022-1-5', freq='B')
pd.bdate_range(start='12/27/2021', end='1/08/2022')
## 不规则日期数据格式转换/格式化过程，格式化的好处是可以使用定义好的属性和标准方法库进行统计和计算
pd.to_datetime(['2022-01-31', '2022-02-28', '2022-03-31'])
pd.to_datetime('13141314', unit='s', errors='ignore')
pd.to_datetime([1641281310, 1741281310], unit='s')

pd.to_datetime('2022-4*1', format='%Y-%m*%d').isocalendar()
df1['成交周']=df1['交易日期'].dt.isocalendar().week

# 时间序列Series
## 注意：24:00是非法时间值，有效值是[0:00, 23:59)
## 创建间隔特定频率的时间
pd.date_range('0:00', '23:59', freq='15min')
## 返回不包含日期的时间datetime.time
pd.date_range('0:00', '23:59', freq='15min').time

# 时间间隔（用于计算）
td = pd.Timedelta(7, 'd')
pd.date_range('2022/4/1', '2022-5-1') + td

# 时区
## 生成带时区的时间序列
pd_datetime = pd.date_range(start='4/1/2022', periods=10, tz='UTC')
## 转换时区
pd_datetime.tz_convert(tz='Asia/Shanghai')
```
> [关于freq支持的频率定义](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases)

### 字符串类型.str.的数据处清洗
```
# 去空格
df1.股票简称 = df1.股票简称.str.strip()
# 判断包含、以开头、以结尾
df1.股票简称.str.contains('ST')
df1['股票简称'].str.startswith('中')
df1[df1['股票简称'].str.endswith('科技')]

# 拼接、分割
df1['营业部信息'] = df1['买方营业部'].str.cat(df1['卖方营业部'], sep='-' * 3)
## expand=True代表分割后保留每一列数据，列名为0,1,2...
df1['营业部信息]'.str.split('-', expand=True)

# 整形
df1['股票简称'].str.pad(10, side='right', fillchar='*'),
df1['股票简称'].str.center(10, fillchar=' ')

# 替换
df1['溢价率'].astype('str').str.replace('-','负')
df1['股票描述'].str.replace('([0-9a-zA-Z])\1{1}','另一个字符串')

# 匹配中文
## 注意：'...+'匹配第一个中文连续字符串，'...+$'只匹配最后一个中文连续字符串，并且以该中文字符串结尾
df1['股票简称'].str.extract('([^\x00-\xff]+$)')
df1['股票简称'].str.extract('([\u4e00-\u9fa5]+)', expand=True)
## extractall匹配多次并以MultiIndex的形式输出多次结果
df1['股票简称'].str.extractall('([^\x00-\xff]+)')

# 格式化英文
## 大小写
ser1.str.upper(), ser1.str.lower()
```

## 索引管理
### 修改索引

```
# 修改表头名称
## inplace=True代表直接修改原数据
df1.rename(columns={'溢价率':'溢价率值', '序号':'大宗交易序号', '成交量（万股）':'万股成交量'}, inplace = True)

# 修改索引列
## 设置某列为索引列（原索引列默认成为新的一列数据）
df2 = df1.set_index('交易日期')
## 使用递增序列号的默认索引列（原索引列默认成为新的一列数据）
df2 = df2.reset_index()

# 按照索引排序
df2.sort_index(ascending=False, inplace=True)
## 判断索引是否是递增的（只要不存在任意递减的规律就是True)
df2.index.is_monotonic_increasing
```
### 多级索引MultiIndex
多级索引的本质是在二维数据的基础之上的三维/多维的操作。除了直接定义多级索引外，pandas里面的方法`.groupby`, `.pivot_table`都可以生成多维索引。
```
# 直接定义多级索引
## 序列Series也有索引，因此Series、DataFrame都有多级索引的概念。多级索引定义的时候要将整个矩阵都定义完整，或者使用标准方法index=pd.MultiIndex.from_product()来定义
df_example = pd.DataFrame(
    data = np.random.randint(100, size=(6, 3)),
    index = [['人民币','人民币','人民币','美元','美元','美元'],['M0','M1','M2','M0','M1','M2']],
    columns = ['统计值（亿元）','同比增长(百分比)','环比增长(百分比）']
    )

# 设置原数据列为多级索引
df2 = df1.set_index(['股票代码', '股票简称', '交易日期'])

# 通过多级索引定位数据
## 多级索引有级别有顺序
df2.loc[688777]
df2.loc[688777].loc['中控技术'].loc[:, '成交价格']
df2.loc[688777, '中控技术', '2022-03-30']
## 以下为错误演示
~~ df2.loc['中控技术'] ~~
~~ df2.loc[688777, '2022-03-30'] ~~

# 多级索引排序
df2.sort_index(level=0, ascending=False)
```
## 数据聚合（数据挖掘）
```
# 聚合.groupby
df1.groupby(['交易日期', '股票简称']).count()

# 聚合计算.groupby + .agg
## 系统会根据列数据类型，判断并自动忽略无法进行计算的列数据
## 多级索引并不改变数据量的行数，而数据聚合计算将有同样特征的数据进行指定算法的聚合计算, 改变了数据量的大小, 分组的特征成为新的索引列index
## 聚合计算默认使用np组件对序列的标准计算函数max, min, mean, median, prod, sum, std, var等，也可以传入自定义函数的对象
df1.agg([np.max, 'min'])

df1.groupby('股票代码').agg('mean')
## 根据特定列进行聚合计算
df1.groupby('股票代码')['成交量（万股）'].agg([np.max, np.sum])
## 根据不同的列进行不同的聚合计算
df1.groupby('股票代码').agg({'成交量（万股）': 'sum', '成交价格': 'median'})

# 聚合转换.groupby + .transform
## 和聚合计算不同的是，聚合转换做了同样类似的数据计算，但不改变数据量的大小，也不改变索引（这也意味着里面有很多duplicated的数据）。计算后的结果返回一个行数相等的序列ser1，可以用方法df1['新列']=ser1定义成原df的新列
df1.groupby('股票代码').成交价格.transform('mean')
df1['成交价波动率'] = df1.groupby('股票简称')['成交价格'].transform(lambda x: (x - x.mean()) / x.std())

df2 = df1.groupby('股票代码')[['成交价格', '最新价']].transform(lambda x: x.astype(int).max())
pd.concat([df1, df2], axis=1)

# 聚合过滤.groupby + .filter
## filter方法需要聚合数据来计算判断。显示的结果不是计算后的数据，而是原数据根据结果过滤后的数据
## 聚合后，根据过滤条件显示符合条件的分组数据，可能会改变数据量的大小，不改变索引值（但过滤后索引不连续、不完整了），不改变列名。
df1.groupby('股票代码').filter(lambda x: x['成交价格'].mean() > 18 and x['成交价格'].mean() < 22)   

# 聚合自定义计算.groupby + .apply
## 自定义的方法，比较灵活
df1.groupby('股票代码').apply(lambda x: x['成交价格'] * x['成交量（万股）'] / x['成交量（万股）'].sum())
df1.groupby('股票代码')['成交量（万股）'].apply(lambda x: x / x.sum())
```

## 数据可视化

- DataFrame和Series都可以后接例如.plot.bar()来画图
- 

### 中文字体
```
# matplotlib的文件路径。font目录下为支持的字体
import matplotlib
matplotlib.matplotlib_fname()

# 操作系统环境内安装中文字体
## fc-list命令查看字体的标准名称
apt-get install fonts-wqy-zenhei
fc-list

# 配置matplot使用指定的中文字体
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
mpl.rcParams['axes.unicode_minus'] =  False
```

### 可视化图形种类
```
# 柱状图 .plot.bar()
df1.head().set_index('股票简称').最新价.plot.bar()

## x不指定默认为数据的index, y不指定默认为数据的所有columns。x和y都可以指定为特定的columns，用于一列数据和另一列数据相对的比较
## rot=0代表坐标轴上的标签为横排格式
## stacked=True代表多列数据叠加在一个条中显示
df2 = df1.set_index('股票简称')[['最新价', '成交量（万股）']]
df2.plot.bar(x='最新价', y='成交量（万股）', rot=0, stacked=True)

## subplots=True代表多列数据分成多个图显示
## color={}指定颜色
df2.plot.bar(subplots=True, color={'最新价': 'red', '成交量（万股）': 'green'})

## title=指定图的标题
## figsize=()指定图大小
df2.head().plot.line(title='趋势图', figsize=(20, 10))

## 保存最近的一张图到当前文件夹的本地文件；查看最近一张保存的图片
plt.savefig('大宗交易趋势图.png')
plt.show()

# 条状图 .plot.barh()
df2.plot.barh()
# 折线图 .plot.line()
df2.plot.line()
# 面积图 .plot.area()
df2.plot.area()
# 饼图 .plot.pie()
## 饼图只显示一列数据
df2.plot.pie(subplot=True)
# 散点图 .plot.scatter()
## 散点图需要指定x和y分别是哪列数据
df2.plot.scatter(x='最新价', y='成交量（万股）')

```
## 辅助库
### numpy
```

# 标准空值，占位符号
np.na

# 0到1之间的随机数一个
np.random.random()

# 随机生成一个一维序列，指定长度。（无索引列）
np.random.randn(1000)
# 随机生成一个二维矩阵，指定行列的长度。（无索引列，无表头）
np.random.randn(10, 3)
# 随机生成一个二维整数矩阵，整数的最大值在100以内
np.random.randint(100, size=(10, 3))

# 定义一个数组/矩阵
np.array([[1, 2], [3, 4]])
np.array([1, 2, 3], dtype=complex)

# 生成起止点之间平均分配的默认50个数，默认包括起止点。等差数列
np.linspace(0, 10)
np.linspace(1, 100, 5)

# 计算一个array里面数据的和，可指定按照行或者按照列，或等其它参数
np.sum([10, 9], initial=5)

# 计算方法
np.mean, 平均值
np.median, 中位数
np.prod, 乘积
np.sum, 加和
np.std, 标准差
np.var, 均方差
```