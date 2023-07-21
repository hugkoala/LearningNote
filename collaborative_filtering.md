#1 part3 #2 part4 #3 deep dive

GPU不像CPU那麼聰明
如果嘗試將模型擴展到更大的模型，除非你有更昂貴的GPU，不然就會用完 並收到 memory不足
call python gc pytorch empty_cache() 基本上可以不重啟的情況開始
記憶體不足 該做什麼  可以用 grandient accumulation
#1 [5] -> 下面說明
梯度累積的結果在數值上相同嗎 對於特定架構來說 數字上是相同 有種東西叫批量歸一化 他追蹤標準差喊平均值的移動平均值 並以數學上稍微不正確的方式進行 因此你進行批量歸依化  基本上會引入更多的波動性 但由於在數學上並不相同 因此你不一定得到相同結果 
convnext 不使用批量規一化
實踐中為 convnext 添加梯度累積沒有帶來任何問題，因此執行時候不必更改任何參數
經驗法則將批量大小除以2 學習率也除以2 但不幸 並不完美

#2 [6] 假如傳進去3個block fastai不知道哪個是自變亮 哪個是因變量
#2 [8] cross_entropy
cross_entropy excel softmax
#2 [11] 

#3 [5] dot product
collab_filter excel
matrix completion
#3 [18]
dot product 有超過5的，雖然沒有sigmoid 已經很好 但是如果放進sigmoid 會更好
#3 [22]使用sigmoid_range 然後為什麼使用0~5.5 而不是5 因為sigmoid 永遠不會到5 所以為了讓我們最高值大一點
但目前dot product沒有方法能說用戶傾向於給低分或高分
#3 [24] 這時候我們在因子增加了偏移項
#3 [25]為什麼變得更糟，所以可能overfit了
避免overfit的一個經典方法叫做 權重衰減 wight decay 也稱 L2 regularization 當我們計算梯度時，將權重平方和加到loss function
#3 [26] 算損失的目的就是為了採用她的梯度，因為參數平方的梯度就是參數的兩倍，所以我們只需要將梯度加上wd係數*2*參數
#3 [27] 在視覺的fastai應用中，會適當的設定預設值，通常做得不錯，只要使用預設值就好，但在表格或是collaborative filtering，對數據不夠了解 不知道在這裡使用什麼 所以可能嘗試10的幾個倍數 從 0.1 除以10幾次 看哪個結果最好

## Understanding GPU memory usage
```python
df = pd.read_csv(path/'train.csv')
df.label.value_counts()
```
![](https://hackmd.io/_uploads/BJgtkdvqn.png)

選擇最小的，我們只需要337張圖像即可訓練模型，較長時間的訓練不會使用更多內存，因此使用最小的訓練集，我們就能知道這個模型使用多少記憶體
```python
trn_path = path/'train_images'/'bacterial_panicle_blight'
train('convnext_small_in22k', 128, epochs=1, accum=1, finetune=False)
```

基本上呼叫完 gc.collect()和pytorch.cuda.empty_cache()，會讓GPU回到乾淨狀態，而不用重啟kernel
```python
import gc
def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()

report_gpu()
```
![](https://hackmd.io/_uploads/SJVWb_D5h.png)

假如 GPU OOM，會發生什麼事

## What is GradientAccumulation?
將批量大小設定為64/accum，現在應該能用比較小的批次解決記憶體問題，但批次規模越小，批次之前的波動性越大，所以你的學習率都亂了，不太可能為每種大小都找到最佳參數
```python
def train(arch, size, item=Resize(480, method='squish'), accum=1, finetune=True, epochs=12):
    dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, item_tfms=item,
        batch_tfms=aug_transforms(size=size, min_scale=0.75), bs=64//accum)
    cbs = GradientAccumulation(64) if accum else []
    learn = vision_learner(dls, arch, metrics=error_rate, cbs=cbs).to_fp16()
    if finetune:
        learn.fine_tune(epochs, 0.01)
        return learn.tta(dl=dls.test_dl(tst_files))
    else:
        learn.unfreeze()
        learn.fit_one_cycle(epochs, 0.01)
```


這是前幾節課使用的訓練迴圈
```python
for x,y in dl:
    calc_loss(coeffs, x, y).backward()
    coeffs.data.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()
```

因為記憶體問題，我必須使用規模較小的批次大小，但又不想為每種都建立一個最佳參數，而想維持每64筆更新一次權重

記得前面呼叫完 backward()
fastai會自動幫我們把梯度加進參數裡，假如沒有清空，再呼叫一次，則會加進原本的梯度裡，透過這種方式，我們就能得到64個批量大小的總梯度，但一次只傳遞32張圖片
```python
count = 0            # track count of items seen since last weight update
for x,y in dl:
    count += len(x)  # update count based on this minibatch size
    calc_loss(coeffs, x, y).backward()
    if count>64:     # count is greater than accumulation target, so do weight update
        coeffs.data.sub_(coeffs.grad * lr)
        coeffs.grad.zero_()
        count=0      # reset count
```
Q:梯度累積的結果在數值上相同嗎?
A:在這個特定的架構來說，數字是相同的
batch normalization follow 標準差和平均的滾動平均值以一種數學上稍微不正確的方式進行，因此進行了batch normalization，基本上會有更多波動，，但這不一定是壞事，因為在數學上不相同，因此你不會得到相同結果，Convnext不使用batch normalization，因此梯度累積數值是相同的

經驗法則，如果將批輛大小除以2，則學習率也除以2，但不幸的是，他不完美

在fastai上很簡單的設定梯度累積，設定要的批次大小=64//accum
呼叫GradientAccumulation(batch size)，得到callback，將callback傳入learner，則獲得batch size，learaner 才會更新權重

```python
train('convnext_small_in22k', 128, epochs=1, accum=4, finetune=False)
report_gpu()
```
![](https://hackmd.io/_uploads/rJuisdv9n.png)

dataloader沒有seed 參數，所以每次都使用不同訓練集
```python
dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, item_tfms=item,
        batch_tfms=aug_transforms(size=size, min_scale=0.75), bs=64//accum)
```


## Multi-target models
建立一個能預測疾病也能預測水稻類型
```python
df = pd.read_csv(path/'train.csv', index_col='image_id')
df.head()
```
![](https://hackmd.io/_uploads/rJxfbtDcn.png)

因為df index_col為image_id，所以能透過loc取到該筆資料，並顯示欄位，同理get_variety 也是這樣取值
```python
df.loc['100330.jpg', 'variety']

def get_variety(p): return df.loc[p.name, 'variety']
```
![](https://hackmd.io/_uploads/H1jxQFD93.png)


為了要建立能預測疾病和水稻類型的模型，首先需要具有兩個因變量的data loader
```python
dls = DataBlock(
    blocks=(ImageBlock,CategoryBlock,CategoryBlock),
    n_inp=1,    # 告訴DataBlock有幾個輸入
    get_items=get_image_files,
    get_y = [parent_label,get_variety],
    splitter=RandomSplitter(0.2, seed=42),
    item_tfms=Resize(192, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75)
).dataloaders(trn_path)
```

```python
dls.show_batch(max_n=6)
```
![](https://hackmd.io/_uploads/HJuUVtD53.png)

多了疾病和類型，所以不能只使用錯誤率當成指標
```python
def disease_err(inp,disease,variety): return error_rate(inp,disease)
```

## What does `F.cross_entropy` do
前面train()中，我們一直都沒討論到使用的損失函數，vision_learner猜測要使用什麼損失函數，發現因變量是單個類別，他知道具有單個類別可能出現的最佳損失函數，也知道該類別有多大，所以learner自動選擇了損失函數，當因變量是類別時，MSE MAE都不能作用
```python
 learn = vision_learner(dls, arch, metrics=error_rate, cbs=cbs).to_fp16()
```

softmax 為觸發函數，確保輸出都界在0和1之間，但跟sigmoid不同，softmax還會確保輸出總合為1
[Softmax and cross-entropy](https://docs.google.com/spreadsheets/d/1r22UD-HX5Xw0WWy5eoZ9qEX2u3QT6GDf/edit?usp=sharing&ouid=101736297677687618260&rtpof=true&sd=true)

1.將模型的輸出(output)轉成機率
2.使用EXP()
3.加總各輸出的EXP值
4.將每個輸出的EXP值除以總EXP值(原輸出較大的值，EXP值也會越大，這個值會被推到更接近1，會像是真的在嘗試選擇，因為它具有最大的機率)
5.在正確的事物實際值為1，其他為0
6.比較softmax和實際值，如果softmax高，實際值高，我們的損失就越小
7.計算各類別 實際值*log(預測值)

## How to calculate binary-cross-entropy

假如我們只關心疾病，只需要計算輸入和疾病的cross entropy
data loader回傳多個目標，fastai不知道實際我們model需要幾個輸出，所以必須指定n_out=10
```python
def disease_loss(inp,disease,variety): return F.cross_entropy(inp,disease)

arch = 'convnext_small_in22k'
learn = vision_learner(dls, arch, loss_func=disease_loss, metrics=disease_err, n_out=10).to_fp16()
lr = 0.01
learn.fine_tune(5, lr)
```
![](https://hackmd.io/_uploads/r19oOivcn.png)


## How to create a learner for prediction two targets

```python
learn = vision_learner(dls, arch, n_out=20).to_fp16()
```

input 會有20列
用了前10列預測疾病
```python
def disease_loss(inp,disease,variety): return F.cross_entropy(inp[:,:10],disease)
```

用了10列以後預測品種
```python
def variety_loss(inp,disease,variety): return F.cross_entropy(inp[:,10:],variety)
```

```python
def combine_loss(inp,disease,variety): return disease_loss(inp,disease,variety)+variety_loss(inp,disease,variety)
```

```python
def disease_err(inp,disease,variety): return error_rate(inp[:,:10],disease)
def variety_err(inp,disease,variety): return error_rate(inp[:,10:],variety)

err_metrics = (disease_err,variety_err)
all_metrics = err_metrics+(disease_loss,variety_loss)
```

```python
learn = vision_learner(dls, arch, loss_func=combine_loss, metrics=all_metrics, n_out=20).to_fp16()
learn.fine_tune(5, lr)
```
![](https://hackmd.io/_uploads/Bkh9Oovq3.png)

## Collaborative filtering deep dive
```python
path = untar_data(URLs.ML_100k)
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])
ratings.head()
```
![](https://hackmd.io/_uploads/BJWJgnP9n.png)

![](https://hackmd.io/_uploads/HkBsghv5h.png)

```python
# the categories are science-fiction, action, and old movies
last_skywalker = np.array([0.98,0.9,-0.9])
user1 = np.array([0.9,0.8,-0.6])
```
我們認為該用戶會多喜歡last sky walker
```python
(user1*last_skywalker).sum()
```
user1 可能會喜歡last sky walker

![](https://hackmd.io/_uploads/Hk0iZ3w5n.png)

```python
casablanca = np.array([-0.99,-0.3,0.8])
(user1*casablanca).sum()
```
user1可能不會喜歡casablanca

![](https://hackmd.io/_uploads/BkzeMhP93.png)

## What are latent factors?

[Collaborative filterings and embeddings](https://docs.google.com/spreadsheets/d/1HrSxTA3izmAAzWRQRtotLtiTLGqRyX2Q/edit?usp=sharing&ouid=101736297677687618260&rtpof=true&sd=true)

1.假設電影有5個潛在因素，但不知道用途，給隨機數
2.用戶也有相同的5個潛在因素，同時也給隨機數
3.假如沒有評分，點積設為0

```python
movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
movies.head()
```
![](https://hackmd.io/_uploads/Hkv4DnDqn.png)

```python
ratings = ratings.merge(movies)
ratings.head()
```
![](https://hackmd.io/_uploads/S1SLv2D53.png)

默認情況下，用戶列稱為user，所以這裡不用代入，而指定項目為title
```python
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()
```
![](https://hackmd.io/_uploads/HkD3D3v5h.png)

## What is embedding
```python
n_users  = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)
```
第3個元素設成1，其他為0
這裡用one-hot encoded vector取第n個值
假如我要取第4個值，我可以設定一個第4個位置為1，其他位置為0的one-hot vector，這樣算完點積後就會得到4
1 2 3 4 5 . 0 0 0 1 0 = 4

embedding:乘以一個one-hot矩陣，藉著直接進行檢索來實作計算捷徑，乘以one-hot矩陣的東西稱為embedding矩陣
```python
one_hot_3 = one_hot(3, n_users).float()
user_factors.t() @ one_hot_3
```
![](https://hackmd.io/_uploads/HJumY3P92.png)

```python
user_factors[3]
```
![](https://hackmd.io/_uploads/HJumY3P92.png)

## How to understand the `forward` function
pytorch將在你的class 呼叫 forward()去做計算
```python
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return (users * movies).sum(dim=1)
```

```python
model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())

learn.fit_one_cycle(5, 5e-3)
```
![](https://hackmd.io/_uploads/rJQcA3vqn.png)

## Adding a bias term

在excel中，我們發現有些值超過最大值
這裡使用了sigmoid_range，但如果你設定0~5，輸出的值永遠不會有5，所以想讓範圍比我們的最高值大一點
```python
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return sigmoid_range((users * movies).sum(dim=1), *self.y_range)
```

```python
odel = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```
![](https://hackmd.io/_uploads/ByRLlpPq3.png)

## Model interpretation
user 29 只是喜歡電影，幾乎所有的電影都給了高分
目前模型沒有任何方法說明用戶傾向於給低分或給高分

```python
class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)
```

```python
model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```
![](https://hackmd.io/_uploads/rkUAWpv92.png)

## What is weight decay and How does it help
計算梯度時，將權重平方和加到損失函數
![](https://hackmd.io/_uploads/rylGSawc2.png)

```python
loss_with_wd = loss + wd * (parameter**2).sum()

# 計算損失就是為了計算梯度，parameters平方和的梯度就會是2 * paramters
parameters.grad = += wd * 2 * parameters
```

```python
model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)
```
![](https://hackmd.io/_uploads/SkjvE6Pc3.png)
