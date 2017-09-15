# 天池智慧交通预测挑战赛解决方案

本博客分享新人第一次参加天池比赛的实况记录，比较完整地给出了数据预处理，缺失值补全，特征分析过程以及训练和交叉验证的注意事项，适合数据挖掘新人找到解题思路，全程没有调参，没有模型融合，只凭一手简单的特征和xgboost，最后止步41/1716，基本上可以作为时间序列预测类的比赛的baseline．完整代码在[Github](https://github.com/PENGZhaoqing/TimeSeriesPrediction)

（ps. 不是我不调参，不融合模型，是以现在的特征即使做了这些，提高也不会很大，所以还是特征的问题，可能太简单了）

* preprocess.py: 数据预处理（类型转换，缺失值处理，特征提取）
* xgbosst.py: 训练模型和交叉验证

# 1. 数据和题目说明

这个比赛的目标是提供一些路段流量的历史信息, 以此来预测未来一段时间的交通流量, 提供的数据一共有3个表: link_info, link_tops 和travel_time. 分别如下所示:

```
link_infos = pd.read_csv('../raw/gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
print link_infos.head(5)

               link_ID  length  width  link_class
0  4377906289869500514      57      3           1
1  4377906284594800514     247      9           1
2  4377906289425800514     194      3           1
3  4377906284525800514     839      3           1
4  4377906284422600514      55     12           1
```

link_info表里共存着每条路的id, 长度, 宽度和类型,共有132条路

```
link_tops = pd.read_csv('../raw/gy_contest_link_top.txt', delimiter=';', dtype={'link_ID': object})
print link_tops.head(5)

               link_ID                                 in_links            out_links
0  4377906289869500514                      4377906285525800514  4377906281969500514
1  4377906284594800514                      4377906284514600514  4377906285594800514
2  4377906289425800514                                      NaN  4377906284653600514
3  4377906284525800514                      4377906281234600514  4377906280334600514
4  4377906284422600514  3377906289434510514#4377906287959500514  4377906283422600514

```
link_top里储存每一条路的上下游关系, in_links里放着这条路的上游路id, 中间用`#`分割, 而out_links里则给出了这条路的下游路id; 下游路可以理解为出路, 上游路为入路
```
df = pd.read_csv('../raw/quaterfinal_gy_cmp_training_traveltime.txt', delimiter=';', dtype={'link_ID': object})
print df.head(5)

               link_ID        date                              time_interval  travel_time
0  4377906283422600514  2017-05-06  [2017-05-06 11:04:00,2017-05-06 11:06:00)         3.00
1  3377906289434510514  2017-05-06  [2017-05-06 10:42:00,2017-05-06 10:44:00)         1.00
2  3377906285934510514  2017-05-06  [2017-05-06 11:56:00,2017-05-06 11:58:00)        35.20
3  3377906285934510514  2017-05-06  [2017-05-06 17:46:00,2017-05-06 17:48:00)        26.20
4  3377906287934510514  2017-05-06  [2017-05-06 10:52:00,2017-05-06 10:54:00)        10.40
```
travel_time表里存着这132条路从2017.4-2017.6以及2016.7每天车通过路的平均旅行时间, 统计的时间间隔为2分钟; 除了2016.4到6月每天的信息, 还有2017.7月每天6:00-8:00, 13:00-15:00, 16:00-18:00的记录, 然后我们需要预测的就是7月每天在早高峰, 午平峰, 晚高峰三个时间段(8:00-9:00, 15:00-16:00, 18:00-19:00)每条路上的车平均旅行时间


# 2. 题目分析和思路

这是一个关于时间序列预测的问题, 并不是普通的回归问题, 而是自回归, 一般的回归问题比如最简单的线性回归模型:`Y=a*X1+b*X2`, 我们讨论的是因变量Y关于两个自变量X1和X2的关系, 目的是找出最优的a和b来使预测值`y=a*X1+b*X2`逼近真实值Y. 而自回归模型不一样, 在自回归中, 自变量X1和X2都为Y本身, 也就是说`Y(t)=a*Y(t-1)+ b*Y(t-2)`,其中`Y(t-1)`为Y在t-1时刻的值, 而 `Y(t-2)`为Y在t-2时刻的值, 换句话说, 现在的Y值由过去的Y值决定, 因此自变量和因变量都为自身, 这种回归叫自回归.

根据题目给出的信息, 除了路本身的信息外, 训练数据基本上只有旅行时间, 而我们要预测的也是未来的平均旅行时间, 而且根据我们的常识, 现在的路况跟过去一段时间的路况是很有关系的, 因此该问题应该是一个自回归问题, 用过去几个时刻的交通状况去预测未来时刻的交通状况

传统的自回归模型有自回归模型（AR）、移动平均模型（MA）、自回归移动平均模型（ARMA）以及差分自回归移动平均模型（ARIMA）, 这些自回归模型都有着严格理论基础,讲究时间的平稳性, 需要对时间序列进行分析才能判断是否能使用此类模型. 这些模型对质量良好的时间序列有比较高的精度, 但此比赛中有大量的缺失值, 因此我们并没有采用此类模型. 我们的思路其实很简单: 就是构建`Y(t)=a*Y(t-1)+ b*Y(t-2)+..`, 但并不是用的线性模型, 用的是基于树的非线性模型, 比如随机森林和梯度提升树.

# 3. 数据分析

## 3.1 特征变换

先对原始数据进行一些分析, 首先了解一下平均旅行时间的分布:

```
fig, axes = plt.subplots(nrows=2, ncols=1)
df['travel_time'].hist(bins=100, ax=axes[0])
df['travel_time'] = np.log1p(df['travel_time'])
df['travel_time'].hist(bins=100, ax=axes[1])
plt.show()
```

![这里写图片描述](http://img.blog.csdn.net/20170911182135429?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


我们发现这个平均旅行时间变量travel_time是一个长尾分布, 也就是大数值的特别少, 而大部分数据都集中在很小的区域, 如上面的图, 因此做一个log的特征变换, 一般对数变换为ln(x+1), 避免x=0而出现负无穷大, 可以看出经过对数变换后, 数据的分布非常均匀, 类似正态分布, 比较适合模型来处理

## 3.2 数据平滑

即使做了log变换后, 还是有部分travel_time值过于大, 为了消除一些离群点的影响, 我们对travel_time做一个百分位的裁剪clip, 我们把上下阈值设为95百分位和5百分位, 即将所有大于上阈值的travel_time归为95百分位数, 而小于小阈值的travel_time设为05百分位数:

```
def quantile_clip(group):
    group.plot()
    group[group < group.quantile(.05)] = group.quantile(.05)
    group[group > group.quantile(.95)] = group.quantile(.95)
    group.plot()
    plt.show()
    return group

df['travel_time'] = df.groupby(['link_ID', 'date'])['travel_time'].transform(quantile_clip)
```

这个clip百分位过滤是对于每一条路每天的travel_time来说的,因此使用groupby方法并传入link_ID和date, 因此得到的group变量里储存着某条路某天的所有travel_time, 对此进行百分位的clip, 如下图:

![这里写图片描述](http://img.blog.csdn.net/20170911185001367?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 


这是Link_ID为4377906283422600514的路在2017-05-06的travel_time时间序列图, 我们clip掉了蓝色的部分, 留下为橙色部分所示

## 3.3 缺失值补全

###  3.3.1 为什么要补全缺失值? 不补全可不可以?

如果我们的思路是通过前几个时刻的travel_time来预测下一个时刻的travel_time, 那么这个缺失值一定要补齐, 因为我们总不能让前几个时刻的travel_time出现空值吧, 而且这种情况下要预测的travel_time也有可能是空值. 如果是其他思路, 比如你的feature中没有出现前几个时刻的travel_time, 而是以minute_of_hour, hour_of_day, day_of_week, day_of_month, month_of_year来标识目前要预测的travel_time的话, 不补全至少模型还是可以跑通的, 至于效果肯定还是补齐后要好点

### 3.3.2 缺失值补全的方法

通常缺失值补齐的方式有中位数，众数，平均数等，在此题目中，缺失值可以被这条路当天所有travel_time的中位数替代，更具体说可以细化到哪个小时的中位数；第二种做法是做插值（interpolate），以现在已有的数据拟合一条曲线或者折线，然后缺失值在就取在这条拟合的线上，具体的有线性插值和样条曲线插值，这种方法比前一种效果要好一些，毕竟插值考虑了缺失值附近的数据，但是问题是：如果这段时间序列有大量的缺失值（或者全是缺失值），那么现在用插值就不合适了，因为拟合的曲线无法反应真实的情况，这种情况下可以用统计历史的情况来解决，也就是用前一天或者几天的这段时间的中位数来补齐缺失值．

前面的方法都是handcraft方法，纯手动补全．那么自动补全的方法就是用预训练一个模型去补全缺失值：训练已有的数据，把缺失的数据当做是要预测的数据，有很多模型可以补全缺失值，例如随机森林等，只要feature构造合理，这种补全的方法要比前面提到的手动的方法效果要好一些，但是不会接近理论值，如果你构造的feature能完美的还原这些缺失值原来的样子，那么补全已经没有意义了，因为现在无论是缺失值还是未来值，你都能达到100%预测了，那你还补齐数据干嘛．那么我们这里采用训练模型的方法来预测缺失值

### 3.3.3 准备工作：找到缺失值

什么？缺失值还用找？不是原来数据集就给你了吗？在此数据集中，缺失的时间序列是没有出现在数据集中的，需要你手动插入数据，然后把travel_time用nan标识出来，所以需要找到这些缺失值，通过pandas的merge方法能很容易实现：先申明完整的dare_range，然后与所有的link_ID进行笛卡尔积，得到的就是一个完整的数据集，这个数据集所有的travel_time都是空值nan，再用这个完整数据集与提供的数据集做表连接（left join），那么原来已有的travel_time就添加到表里，缺失的值还是nan.

```
link_df = pd.read_csv('../raw/gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})

date_range = pd.date_range("2016-07-01 00:00:00", "2016-07-31 23:58:00", freq='2min').append(
    pd.date_range("2017-04-01 00:00:00", "2017-07-31 23:58:00", freq='2min'))

new_index = pd.MultiIndex.from_product([link_df['link_ID'].unique(), date_range],
                                       names=['link_ID', 'time_interval_begin'])
df1 = pd.DataFrame(index=new_index).reset_index()
df3 = pd.merge(df1, df, on=['link_ID', 'time_interval_begin'], how='left')
```
标示出缺失值后，先看一下缺失值的分布情况，这里只选择了每天6, 7, 8, 13, 14, 15, 16, 17, 18小时的数据（因为与要预测的就是8,15,18小时），然后去掉2017年7月每天第8,15,18小时，这部分是我们的预测值，不属于缺失值，这样我们画出每天缺失值的个数如下：

```
df3 = df3.loc[(df3['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]
df3 = df3.loc[~((df3['time_interval_begin'].dt.year == 2017) & (df3['time_interval_begin'].dt.month == 7) & (
    df3['time_interval_begin'].dt.hour.isin([8, 15, 18])))]

df3['date'] = df3['time_interval_begin'].dt.strftime('%Y-%m-%d')

df3.loc[df3['travel_time'].isnull() == True].groupby('date')['link_ID'].count().plot()
plt.show()
```

![这里写图片描述](http://img.blog.csdn.net/20170913153243869?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

根据此图，我们发现2016年7月的缺失值比较多，这点就反应了这个月的数据质量不如其他几个月好，其实后面训练模型做交叉验证的时候就发现对7月的预测效果很不好，实际上2016年7月到2017年4月中间还是有很大一个gap的，搞不好中间就发生了修路，原来2017年很堵的路现在不堵了，这点在后面的分析中也得到了验证，后面会介绍．事实上，我在这里把2017年7月的数据换成了2017年3月的数据（初赛的数据），这样就变成了用2017年的3,4,5,6四个月来预测7月

### 3.3.4 补全步骤：Seasonal date trend+ Daily hour trend + xgboost predict

**1.Seasonal date trend**: 首先从大的趋势来看，车流量应该是有个季节性的变化的，我们第一步研究研究这个季节性的变化，我们针对每一条路，画出它从3月到7月的travel_time变化，时间粒度单位为小时，也就是3月到7月每个小时travel_time的变化量，如下代码：

```
def date_trend(group):
    tmp = group.groupby('date_hour').mean().reset_index()

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    y = tmp['travel_time'].values
    nans, x = nan_helper(y)
    if group.link_ID.values[0] in ['3377906282328510514', '3377906283328510514', '4377906280784800514',
                                   '9377906281555510514']:
        tmp['date_trend'] = group['travel_time'].median()
    else:
        regr = linear_model.LinearRegression()
        regr.fit(x(~nans).reshape(-1, 1), y[~nans].reshape(-1, 1))
        tmp['date_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
    group = pd.merge(group, tmp[['date_trend', 'date_hour']], on='date_hour', how='left')
    plt.plot(tmp.index, tmp['date_trend'], 'o', tmp.index, tmp['travel_time'], 'ro')
    plt.title(group.link_ID.values[0])
    plt.show()
    return group


df['date_hour'] = df.time_interval_begin.map(lambda x: x.strftime('%Y-%m-%d-%H'))
df = df.groupby('link_ID').apply(date_trend)
df = df.drop(['date_hour', 'link_ID'], axis=1)
df = df.reset_index()
df = df.drop('level_1', axis=1)
df['travel_time'] = df['travel_time'] - df['date_trend']

```
给出其中的一个图，这是link_id为3377906280028510514的路从3月到7月每个小时travel_time的变化，红色点表示真实数据，蓝色线表示我们对这个趋势的回归，这里用的是线性回归，也可以用曲线来回归，例如spline，但是这里好像曲线特征不怎么明显，从这个图可以看出这条路从3月到7月的travel_time大致呈稍微上升趋势，没有红色点的地方说明这块就是缺失的值，我们希望拟合的蓝线能对这部分的缺失的值的走向进行一个大致预测：

![这里写图片描述](http://img.blog.csdn.net/20170913170529101?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

有些路缺失值比较严重，比如下图，大概从5月到6月一个数据都没有，这个时候还用线性回归可能会对时间上比较远的数据造成比较大的误差，为了保守起见，这时蓝线用一条平行x轴的直线代替，表示这条路的travel_time未来走向不变，蓝线与y轴的交点为前面已知点的中位数

![这里写图片描述](http://img.blog.csdn.net/20170913171617012?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

如果我们使用2016年7月的数据的话，会出现下面的情况，可以看出来前面2016年7月与2017年4,5,6月的数据有明显断层，原因可能是路翻修了，因此2016年7月的数据对2017年的预测没有很大的作用，改成了2017年3月的数据．当然并不是所有的路都是这样，只有三分之一的路有这样大的变化，因此你也可以手动找出这些路，然后去掉这些路2016年7月的数据，剩下的数据也可以拿来预测

![这里写图片描述](http://img.blog.csdn.net/20170913172352193?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


> NOTE: 我们将蓝线回归得到的值存在`df['date_trend']`里，因此现在的travel_time就更新为`df['travel_time'] = df['travel_time'] - df['date_trend']`，表示date_trend作为大的趋势已经被线性回归决定了，剩下的就是研究这个残差了，之后我们训练和预测都是基于残差，最后用预测出来的残差加上相应的date_trend即可得到需要的预测值


**2.Daily hour trend** :　前面我们获得了季节性的残差，那么我们还可以构造更细的残差，也就是daily hour trend，我们可以观察车流量在一天里的变化趋势，我们针对每一条路，画出它从第6,7,8,13,14,15,16,17,18小时的travel_time变化，时间粒度单位为分钟，此时travel_time为这条路从3月到7月每天在这个时间点（分钟）的平均值，如下代码:

```
def minute_trend(group):
    tmp = group.groupby('hour_minute').mean().reset_index()
    spl = UnivariateSpline(tmp.index, tmp['travel_time'].values, s=1, k=3)
    tmp['minute_trend'] = spl(tmp.index)
    plt.plot(tmp.index, spl(tmp.index), 'r', tmp.index, tmp['travel_time'], 'o')
    plt.title(group.link_ID.values[0])
    plt.show()
    # print group.link_ID.values[0]
    group = pd.merge(group, tmp[['minute_trend', 'hour_minute']], on='hour_minute', how='left')

    return group

df['hour_minute'] = df.time_interval_begin.map(lambda x: x.strftime('%H-%M'))
df = df.groupby('link_ID').apply(minute_trend)

df = df.drop(['hour_minute', 'link_ID'], axis=1)
df = df.reset_index()
df = df.drop('level_1', axis=1)
df['travel_time'] = df['travel_time'] - df['minute_trend']
```

下图给出其中link_id为3377906280028510514的路在一天之内travel_time的变化走势，这个走势综合了所有3月到7月每天的travel_time，因此能被看做是这条路一天里比较普适的情况，而且这里我们具体到了分钟，因为如果只具体到hour数据点太少．下图中蓝色点为每分钟的travel_time数据，而红线为我们为此回归的一条曲线来表示这个走势，使用的是样条曲线拟合：

![这里写图片描述](http://img.blog.csdn.net/20170913181341926?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

有些路的走势变化比较大，需要修改UnivariateSpline中s的参数来决定拟合的程度，s的值越小，对数据拟合越好，但是会过拟合的危险；s的值越大，拟合条件越宽松，下面两个图第一个图的s值为1，第二个图的s值为0.2，请对比两个图的区别，因为不同的路的变化不一样，所以s的值对于不同的路会变化，这个需要手动调整：


![这里写图片描述](http://img.blog.csdn.net/20170913182444336?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20170913182341859?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

> NOTE: 与季节残差一样，我们将红线回归得到的值存在`df['minute_trend']`里，因此现在的travel_time再次更新为`df['travel_time'] = df['travel_time'] - df['minute_trend']`，表示hour_trend作为一天之内走势的已经被样条曲线决定了，那么这个残差加上`df['minute_trend']`和之前的`df['date_trend']`就能还原预测值，我们继续研究这个残差

**3.Xgboost predict**: 基本上大概的走势已经被date_trend和hour_trend决定了，剩下就是研究这个travel_time如何围绕这两个trends上下变化的，我们使用非线性的xgboost来训练，关于时间的feature非常简单，基本上为minute, hour, day, week_day, month, vacation, 其他关于的路本身的feature后面再讲，训练的数据train_df 为travel_time非空的数据，而测试集test_df为travel_time空的数据，训练好后的模型能直接将这些空的数据预测出来并储存在`test_df['prediction']`里，最后与原来的数据合并．我们这里使用`df['imputation1']`标记出这个travel_time是原数据还是后来补全的数据，以便于查看补全的效果．

```
link_infos = pd.read_csv('raw/gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
link_tops = pd.read_csv('raw/gy_contest_link_top.txt', delimiter=';', dtype={'link_ID': object})
link_tops['in_links'] = link_tops['in_links'].str.len().apply(lambda x: np.floor(x / 19))
link_tops['out_links'] = link_tops['out_links'].str.len().apply(lambda x: np.floor(x / 19))
link_tops = link_tops.fillna(0)
link_infos = pd.merge(link_infos, link_tops, on=['link_ID'], how='left')
link_infos['links_num'] = link_infos["in_links"].astype('str') + "," + link_infos["out_links"].astype('str')
link_infos['area'] = link_infos['length'] * link_infos['width']
df = pd.merge(df, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'], how='left')

df.loc[df['date'].isin(
    ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
     '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1

df.loc[~df['date'].isin(
    ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
     '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

df['minute'] = df['time_interval_begin'].dt.minute
df['hour'] = df['time_interval_begin'].dt.hour
df['day'] = df['time_interval_begin'].dt.day
df['week_day'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
df['month'] = df['time_interval_begin'].dt.month

def mean_time(group):
    group['link_ID_en'] = group['travel_time'].mean()
    return group

df = df.groupby('link_ID').apply(mean_time)
sorted_link = np.sort(df['link_ID_en'].unique())
df['link_ID_en'] = df['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))

def std(group):
    group['travel_time_std'] = np.std(group['travel_time'])
    return group

df = df.groupby('link_ID').apply(std)
df['travel_time'] = df['travel_time'] / df['travel_time_std']


params = {
    'learning_rate': 0.2,
    'n_estimators': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'max_depth': 10,
    'min_child_weight': 1,
    'reg_alpha': 0,
    'gamma': 0
}

df = pd.get_dummies(df, columns=['links_num', 'width', 'minute', 'hour', 'week_day', 'day', 'month'])

print df.head(20)

feature = df.columns.values.tolist()
train_feature = [x for x in feature if
                 x not in ['link_ID', 'time_interval_begin', 'travel_time', 'date', 'travel_time2', 'minute_trend',
                           'travel_time_std', 'date_trend']]

train_df = df.loc[~df['travel_time'].isnull()]
test_df = df.loc[df['travel_time'].isnull()].copy()

print train_feature
X = train_df[train_feature].values
y = train_df['travel_time'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

eval_set = [(X_test, y_test)]
regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                             booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                             colsample_bytree=params['colsample_bytree'], random_state=0,
                             max_depth=params['max_depth'], gamma=params['gamma'],
                             min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_set=eval_set)

test_df['prediction'] = regressor.predict(test_df[train_feature].values)

df = pd.merge(df, test_df[['link_ID', 'time_interval_begin', 'prediction']], on=['link_ID', 'time_interval_begin'],
              how='left')

feature_vis(regressor,train_feature)

df['imputation1'] = df['travel_time'].isnull()
df['travel_time'] = df['travel_time'].fillna(value=df['prediction'])
df['travel_time'] = (df['travel_time'] * np.array(df['travel_time_std']) + np.array(df['minute_trend'])
                     + np.array(df['date_trend']))
```

> NOTE: 　经过前两个trend的相减变换，现在的travel_time基本上在0附近波动，均值为0, 我们还可以把算出每条路的travel_time的标准差 `df['travel_time_std']`，用`df['travel_time'] = df['travel_time'] / df['travel_time_std']`来b标准化每条路的travel_time的方差为1，这种均值为0，方差为1的数据分布对于模型来说是比较理想的状态（特别是深度学习）

至此，缺失值补全就结束了，我们可以看一下补全的效果, 我们画出某路某天的travel_time变化如下图，红色的部分是补全的数据，蓝线为原来的数据，可以看出补全的数据比较保守，基本贴近于hour_trend：

```
def vis(group):
    group['travel_time'].plot()
    tmp = group.loc[group['imputation1'] == True]
    plt.scatter(tmp.index, tmp['travel_time'], c='r')
    plt.show()

df.groupby(['link_ID', 'date']).apply(vis)
```

![这里写图片描述](http://img.blog.csdn.net/20170913203827959?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


## 3.4 分析提取特征

这次比赛用的特征主要是分为时间相关的特征和每条路本身的特征，时间特征主要是前几个时刻的travel_time, 这个被叫做lagging，然后是hour和day_of_week, 与路相关的特征为路的长度，宽度，id，路的类型．选特征是有一定的依据的，应该是这个特征的存在对于区分预测值有贡献才会入选，我们先看一下路的特征：

### 3.4.1 与路相关的特征

**１.基本特征(长度，宽度)：**像路的长度，宽度这种特征肯定是直接影响车通行的时间的．常识告诉我们，路越长，车需要跑完这段路需要的时间越长；路越宽，车跑起来越快，那么通行时间越短，那我们看一下是不是这么回事，我们根据车的长度和宽度画出travel_time的箱线图(盒图)如下：

![这里写图片描述](http://img.blog.csdn.net/20170915130201906?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

毫无疑问，路的长度与travel_time是成正比的，路越长，travel_time越大，所以路的长度特征应该是非常重要的，图中黑色的圈是箱线图中的离群点，表示这些点处于1.5倍的四分位极差（IQR）之外，这些离群点大部分正是堵车造成的，导致travel_time有时候异常的高

![这里写图片描述](http://img.blog.csdn.net/20170915130512744?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

反观路的宽度，随着路的宽度的增大，车的通行时间并没有很明显的增大，反而在宽度为12米的时候有所下降，看来与常识有点出入，但是区分度还是有的，因此也可以作为特征之一

**2.其他特征(上下游关系，ID特征)**：此题给了我们这132条路的上下游关系，那么对于每一条路来说, 基本上都有上游和下游，但是有的路处于尽头，本身就没有上游或者下游，我们这里只根据上游和下游的数量进行划分，统计每一条路的上游和下游路的个数，然后画出箱线图：

![这里写图片描述](http://img.blog.csdn.net/20170915134325881?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

上图的in_links和out_links分别表示的上游路和下游路的数量，可以看出根据in_links和out_links组合方式划分对travel_time来说还是比较有区分度的，因此这个特征是有必要的，表示了路的类型．

那么ID特征呢，ID作为唯一标识一条路的指示符，应该也能作为特征之一，但是ID为类别特征，不能直接输入到模型里，一般需要onehot处理，而且onehot后，特征维度会增加132维，计算量会极大增加，对于树模型来说，一个类别特征onehot后，这个类别的重要性会极大的下降，特别是随机森林

> 随机森林中的每一棵树对原来特征都是进行随机选取的，那么onehot之后，这个类别特征变得非常稀疏，每一颗树只能选取到这个类别中的一小部分，远不如直接完整选择这个类别变量；对于xgboost来说还好，虽然它也有一个参数colsample_bytree能控制每次迭代随机选取的feature个数，但是这个值一般在0.6-0.8，而随机森林中这个参数一般处于0.3．同样的特征，分别用xgboost和随机森林训练后，画出各个特征的重要性，会发现xgboost对onehot后特征的重要性要稍微大一些，个人理解，详细请看[这里](https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html)

**那么ID特征不做onehot，用label encode可不可以呢？** scikit库有直接将ID映射到标签的工具，映射后ID特征变为了从1到132离散的数字，首先要明白原来的ID类别是不具有任何顺序和大小的(categorical variables)，而映射后的１到132是orderd，而且是有大小的(numerical variable)．我们知道xgboost和随机森林能够同时处理连续和离散的特征，但是以现在的实现来看，这两者还不具备对把这1到132数字当做类别去看待，因为这两者(scikit的random forest和官方xgboost)对特征划分方式的实现(回归)是基于Numerical的， 与基于信息熵的决策树不同，他们会认为根据id的大小去选择划分点，因此对于想要使用scikit的随机森林和xgboost来说，onehot是必要的，但好像H2O库的实现是可以的，详细请看[这里](https://stackoverflow.com/questions/24715230/can-sklearn-random-forest-directly-handle-categorical-features)

我的做法还是使用label encode, 但是从ID特征到1-132数字的映射我是根据平均的travel_time排序生成的，也就是说现在ID为1的路本身travel_time就很小，而ID为132的确travel_time是最大的，这样这个1-132数字就与travel_time有了比较强的关联，这种编码方式也使这个ID feature有了比较高的重要性，缺点是容易导致过拟合，因为信息总是掌握不完全的，一旦你现在的排序对于未来要预测的结果有很大出入，那么就会有问题．关于类别变量更多的处理，可以看[这里](http://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html)

![这里写图片描述](http://img.blog.csdn.net/20170915150254789?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

以下代码为关于路的特征的生成和可视化：

```
link_infos = pd.read_csv('../raw/gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
link_tops = pd.read_csv('../raw/gy_contest_link_top.txt', delimiter=';', dtype={'link_ID': object})
link_tops['in_links'] = link_tops['in_links'].str.len().apply(lambda x: np.floor(x / 19))
link_tops['out_links'] = link_tops['out_links'].str.len().apply(lambda x: np.floor(x / 19))
link_tops = link_tops.fillna(0)
link_infos = pd.merge(link_infos, link_tops, on=['link_ID'], how='left')
link_infos['links_num'] = link_infos['in_links'] + link_infos['out_links']
link_infos['links_num'] = link_infos["in_links"].astype('str') + "," + link_infos["out_links"].astype('str')
link_infos['area'] = link_infos['length'] * link_infos['width']
df = pd.merge(df, link_infos[['link_ID', 'length', 'width', 'in_links', 'out_links', 'links_num', 'area']],
              on=['link_ID'], how='left')

def mean_time(group):
    group['link_ID_en'] = group['travel_time'].mean()
    return group

df = df.groupby('link_ID').apply(mean_time)
sorted_link = np.sort(df['link_ID_en'].unique())
df['link_ID_en'] = df['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))

df.boxplot(by=['length'], column='travel_time')
plt.show()
df.boxplot(by=['width'], column='travel_time')
plt.show()
df.boxplot(by=['in_links', 'out_links'], column='travel_time')
plt.show()
df.boxplot(by=['link_ID_en'], column='travel_time')
plt.show()

```

### 3.4.2 与时间相关的特征

**1.lagging特征**: 前面分析过，我们的主要目的是通过前几个时刻的travel_time来预测下一个时刻的travel_time，那么就需要构造lagging特征，lagging的个数我们取5,  也就是用lagging1, lagging2, lagging3, lagging4 和lagging5来预测现在t时刻的travel_time，其中lagging1表示t-1时刻的travel_time, 以此类推．通过pandas的表连接操作，我们能很容易构造出来：

```
def create_lagging(df, df_original, i):
    df1 = df_original.copy()
    df1['time_interval_begin'] = df1['time_interval_begin'] + pd.DateOffset(minutes=i * 2)
    df1 = df1.rename(columns={'travel_time': 'lagging' + str(i)})
    df2 = pd.merge(df, df1[['link_ID', 'time_interval_begin', 'lagging' + str(i)]],
                   on=['link_ID', 'time_interval_begin'],
                   how='left')
    return df2

df1 = create_lagging(df, df, 1)
for i in range(2, lagging + 1):
    df1 = create_lagging(df1, df, i)

print df1.head(7)

               link_ID        date time_interval_begin  travel_time  imputation1  lagging1  lagging2  lagging3  lagging4  lagging5
0  3377906280028510514  2017-03-06 2017-03-06 06:00:00         1.63         True       nan       nan       nan       nan       nan
1  3377906280028510514  2017-03-06 2017-03-06 06:02:00         1.61         True      1.63       nan       nan       nan       nan
2  3377906280028510514  2017-03-06 2017-03-06 06:04:00         1.62         True      1.61      1.63       nan       nan       nan
3  3377906280028510514  2017-03-06 2017-03-06 06:06:00         1.64         True      1.62      1.61      1.63       nan       nan
4  3377906280028510514  2017-03-06 2017-03-06 06:08:00         1.67         True      1.64      1.62      1.61      1.63       nan
5  3377906280028510514  2017-03-06 2017-03-06 06:10:00         1.69         True      1.67      1.64      1.62      1.61      1.63
6  3377906280028510514  2017-03-06 2017-03-06 06:12:00         1.72         True      1.69      1.67      1.64      1.62      1.61
```

**2.基本时间特征(week_day, hour, vacation)**:  基本的时间特征有很多，比如minute_of_hour, hour_of_day, day_of_month, day_of_week, month_of_year以及year，这些都是可以挖掘的，经过我的尝试，我只发现day_of_week, hour_of_day以及vacation是比较有用的，根据常识来说，车在工作日是比较多的，travel_time相对大，而在一天之内，上下班高峰期也直接影响travel_time，而且假期也是很影响大家的出行的．我们可以跟上面一样对week_day和hour以及vacation分别画出箱线图看看，但我后来发现了week_day和hour是有一定的关联的，比如周一的早上8点与周末的早上8点是完全不一样的，下面给一天之内每个小时平均travel_time的变化情况

```
df.loc[df['time_interval_begin'].dt.month.isin([3, 4, 5, 6])].groupby(['hour', 'day_of_week'])[
    'travel_time'].mean().unstack().plot()
plt.show()
```

![这里写图片描述](http://img.blog.csdn.net/20170915153506885?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

从上图我们看出, 周6和周7(周日)的变化趋势是相似的，因为它们属于周末；而周1到周5变化趋势相同，因为他们属于工作日．而在周末中，周六的travel_time相对高一些，说明出行在周6更多一些；而在工作日，周4和周5在下班时期travel_time要高一些，说明有一部分人需要往其他地方赶，比如北京的周五下班就有很多人往郊区赶，可以大概推断这些路处于城市区域．既然week_day和hour有关联，那么手动把他们作为一个组合特征比单独两个特征在模型里表现更好. 

![这里写图片描述](http://img.blog.csdn.net/20170915155311094?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

上图为组合后week_hour的箱线图，可能看起来不太明显，但是对比两个单独的特征的箱线图要好一点，感兴趣的可以自己试试，这里我并没有直接将week_day和hour直接组合，根据上面的分析，而是先手动做了个聚类，week里把周1,2,3归为一类，周4,5归为一类，周6,7归为一类；hour里也分三类为：6,7,8(早高峰)，13,14,15(午平峰), 16,17,18(晚高峰)．这么做把组合数量的特征个数从7*9=63压缩到了3*3=9，一定程度上能防止过拟合，而且避免分散了每个组合特征的重要性

```
df2['day_of_week'] = df2['time_interval_begin'].map(lambda x: x.weekday() + 1)
df2.loc[df2['day_of_week'].isin([1, 2, 3]), 'day_of_week_en'] = 1
df2.loc[df2['day_of_week'].isin([4, 5]), 'day_of_week_en'] = 2
df2.loc[df2['day_of_week'].isin([6, 7]), 'day_of_week_en'] = 3

df2.loc[df['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'hour_en'] = 1
df2.loc[df['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'hour_en'] = 2
df2.loc[df['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'hour_en'] = 3

df2['week_hour'] = df2["day_of_week_en"].astype('str') + "," + df2["hour_en"].astype('str')
df2.boxplot(by=['week_hour'], column='travel_time')
plt.show()
```

最后是vacation, 直接0,1对当前日期是否为法定假期进行编码：

```
df2.loc[df2['date'].isin(
    ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
     '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1

df2.loc[~df2['date'].isin(
    ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
     '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0
```

## 4. 训练模模型和Cross Valid

训练模型时直接用上面的feature训练对应的travel_time就可以，但是时间序列feature在交叉验证和测试的时候就不能跟训练一样了，因为当我们在预测出t时刻的travel_time后，需要把这个travel_time作为预测t+1时刻travel_time的lagging1特征，这个lagging特征是需要根据上次预测的结果进行更新的，如此反复直到预测到最后一个时刻的travel_time，然后在划分本地训练集和本地验证集的时候，不能随机划分，标准的话要根据时间顺序来划分，比如现在根据时间顺序将训练集分为5-fold，那么５次交叉验证的训练集和测试集分别为如下：

 - fold 1 : training [1], test [2]
 - fold 2 : training [1 2], test [3]
 - fold 3 : training [1 2 3], test [4]
 - fold 4 : training [1 2 3 4], test [5]
 - fold 5 : training [1 2 3 4 5], test [6]

如果按照一般的CV, 现在有一种情况是training [1 2 4 5 6], test [3]，那么3中出现的trend，有可能被在5和6中重复，如果把5和6拿来训练，那么预测3的效果就会非常好，因此导致模型(特征)被高估，详细请看[这里](https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection)，但是kaggle上也有说可以用一般的CV，有可能是有些训练集在划分后的子训练集相互比较独立，在时间上没有什么联系，因此一般的CV也可以．关于训练和CV的代码，请查看[这里](https://github.com/PENGZhaoqing/TimeSeriesPrediction/blob/master/xgbosst.py)

![这里写图片描述](http://img.blog.csdn.net/20170915205044640?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

最后训练的结果本地CV为0.267324，而线上为0.268095，我们画出各个特征的重要性，可以看出lagging特征是最重要的，说明前几个时刻的travel_time对下一个时刻的travel_time预测的贡献最大．

**1. 如何观察自己线下划分的验证集是否正常？**　线上的结果和线下的结果如果保持同增同减说明你的划分是有效的，但是如果线上和线下结果不同步，其中导致的原因有很多，不一定是验证集划分有误，但最有可能的是过拟合，而且过拟合并不只是模型的问题，还有可能是你选择的特征本来就非常容易过拟合你的训练集而在未知的测试集中表现很差．我在初赛中曾选择过没有onehot的minute特征，当时在线下表现非常好，但是线上炸了，我曾一度怀疑是我的xgboost模型参数过拟合，导致我浪费大量时间去增加正则，在初赛最后几次提交中才意识到这个问题，差点就错过复赛.

**2. Trust your local CV?** 在Kaggle的比赛分享中，基本上很多人都说这句话，个人相信在Kaggle的比赛中这句话是有用的，但是对于天池比赛，这句话就会有一定的误导性了．Kaggle的比赛有public LB和private LB之分，public LB只是测试集的一部分(随机)，当你的结果在public LB中表现不好的时候，有可能并不是你模型或者特征的问题，单纯是public LB的划分不适合．但是天池的线上每次都是测试完整的测试集，因此返回的线上结果一定可靠的，那么Trust your local CV就不再适用了：如果你现在本地表现好而线上爆炸，那么一定是过拟合了．但是，每次用完整的测试集返回结果会导致选手去不断尝试过拟合这个完整的测试集，因此为了保持模型的泛化能力，天池每次在提交最后几次会更换数据重新排名，目的就是要选出效果好而且泛化能力强的模型．我在之前一直相信着Trust your local CV这句话，导致我的本地CV结果超好而线上爆炸，却还特别疑惑：难道不是本地CV效果好了，线上也是一样吗？

**3. 如何根据本地CV判断模型(特征)的好坏？**　一般来说，取5次fold的准确率的平均数是一个判断标准，说明了模型的偏差，另外一个就是这5个准确率的标准差，尽量选择标准差小的一组，标准差小说明模型针对于训练集来说方差小，不易过拟合

**4. 过拟合该怎么办？**　过拟合首先选择检查特征，对于树模型来说，越细分的特征越容易导致过拟合，比如ID，minute这样，这些特征本身包含的值太多，很容易就让数据集划分出大量分区，信息上纯度来说没有问题，对于本地CV效果可能会很好，但是到了线上有很大的可能性会过拟合．就像决策树中ID3的改进版本C4.5，使用了信息增益作为分裂准则，能够很好避免选择这些特征进行划分．然后就是检查模型，一般来说模型默认的参数是不会太容易过拟合的，如果你觉得现在的模型参数已经不可能过拟合了，那么一定是你的特征出问题了．

# 5. 总结

对于此比赛的提高：

 - 第一个需要尝试的就是lagging特征的变化，因为lagging特征是所有特征中重要性最高的，稍微改善lagging特征，可能对结果的提高特别大，关于lagging还可以选择前几个时刻的travel_time的统计特征, 平均值，中位数等等，甚至还可以加上前几天的travel_time的统计特征
 - 然后就是模型的选择，其实树模型不太适合处理时间特征，因为树模型不能很好低预测之前未出现过的情况，这点是硬伤，但是相比其他模型，树模型对结果有着很好的解释，通过分裂过程可以直观了解各个特征的重要性，所以树模型也经常用来选特征

然后给新手的一些建议：

 - 重心一定要放在特征上，garbage in garbage out，特征决定着你的成绩上限，模型只不过是去逼近这个上限，一组好的特征不用非常的复杂的模型和调参就能取得一个比较好的结果，这组特征再上各种模型才是比较值的，不然就是白费力气
 - 对数据本身一定要去详细了解，尽可能去挖掘其中的一些规律，这样自己的弄的特征才会有意义，要知道为什么这个特征会起作用，不然就是耍流氓
 - 在使用模型前，最好了解其中的原理，这样能够更有效的使用，尤其是对类别变量的处理
 - 每次提交的时候，距离上一次的提交不要有太多变化，所谓控制变量，不然效果差都不知道是哪个地方出了问题，有经验的大佬可以忽略这个

