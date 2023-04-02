# Neural net foundations

![](https://i.imgur.com/lGmqTfw.png)

## Which image models are best
[Which image models are best](https://www.kaggle.com/code/jhoward/which-image-models-are-best/)

通常在意
1. 速度多快
2. 使用多少記憶體
3. 準確度多高

### timm
> **PyTorch Image Models (timm) is a wonderful library by Ross Wightman which provides state-of-the-art pre-trained computer vision models**. It's like Huggingface Transformers, but for computer vision instead of NLP (and it's not restricted to transformers-based models)!

> Ross has been kind enough to help me understand how to best take advantage of this library by identifying the top models. I'm going to share here so of what I've learned from him, plus some additional ideas.

```
# Get all models
! git clone --depth 1 https://github.com/rwightman/pytorch-image-models.git
%cd pytorch-image-models/results

# Read csv
'''We'll also add a "family" column that will allow us to group architectures into categories with similar characteristics:

Ross has told me which models he's found the most usable in practice, so I'll limit the charts to just look at these. (I also include VGG, not because it's good, but as a comparison to show how far things have come in the last few years.)
'''

import pandas as pd
df_results = pd.read_csv('results-imagenet.csv')

def get_data(part, col):
    df = pd.read_csv(f'benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv').merge(df_results, on='model')
    df['secs'] = 1. / df[col]
    df['family'] = df.model.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)')
    df = df[~df.model.str.endswith('gn')]
    df.loc[df.model.str.contains('in22'),'family'] = df.loc[df.model.str.contains('in22'),'family'] + '_in22'
    df.loc[df.model.str.contains('resnet.*d'),'family'] = df.loc[df.model.str.contains('resnet.*d'),'family'] + 'd'
    return df[df.family.str.contains('^re[sg]netd?|beit|convnext|levit|efficient|vit|vgg|swin')]

df = get_data('infer', 'infer_samples_per_sec')


import plotly.express as px
w,h = 1000,800

def show_all(df, title, size):
    return px.scatter(df, width=w, height=h, size=df[size]**2, title=title,
        x='secs',  y='top1', log_x=True, color='family', hover_name='model', hover_data=[size])
        
show_all(df, 'Inference', 'infer_img_size')

```

![](https://i.imgur.com/ILFP9XE.png)
X軸為秒數，Y軸為在imagenet的準確度

(ImageNet專案是一個大型視覺資料庫，用於視覺目標辨識軟體研究。該專案已手動注釋了1400多萬張圖像，以指出圖片中的物件，並在至少100萬張圖像中提供了邊框)

resnet18 是用於prototyping 特別小且快的版本，我們經常使用 resnet34

## How to use model for your pet
```
from fastai.vision.all import *
import timm

path = untar_data(URLs.PETS)/'images'

dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=RegexLabeller(pat = r'^([^/]+)_\d+'),
    item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(3)
```
![](https://i.imgur.com/u20cjlX.png)

```
# list all converxt* model
timm.list_models('convnext*')
```
![](https://i.imgur.com/ZbkeA5R.png)

```
# Use your model
learn = vision_learner(dls, 'convnext_tiny_in22k', metrics=error_rate).to_fp16()
learn.fine_tune(3)
```
![](https://i.imgur.com/4ohYbud.png)

```
learn.export('model.pkl')
```

[Convnext documentation](https://course.fast.ai/Lessons/lesson3.html)

### Models trained on ImageNet-22k, fine-tuned on ImageNet-1k
convnext_tiny_in22ft1k
### Models trained on ImageNet-22k, fine-tuned on ImageNet-1k at 384 resolution
convnext_tiny_384_in22ft1k

covnext_{stochastic depth rate}



## Use your model to predict
```
#|export
from fastai.vision.all import *
import gradio as gr
import timm

im = PILImage.create('basset.jpg')
im.thumbnail((224,224))
im
```
![](https://i.imgur.com/rsvLop4.png)
```
#export
learn = load_learner('model.pkl')
learn.predict(im)
# 結果為 37個品種的狗貓概率
```
![](https://i.imgur.com/kv109Xn.png)
```
#export
categories = learn.dls.vocab

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

classify_image(im)
# 將預測各類別結果和列表mapping 
```
![](https://i.imgur.com/Pvvc0sJ.png)
```
#export
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['basset.jpg']

#export
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
```
![](https://i.imgur.com/qIO65gr.png)
```
m = learn.model
m
# model.pkl 有2個主要的部份，分別為預處理的步驟列表和
# 本身由層組成
```
![](https://i.imgur.com/rUgOF0A.png)
```
l = m.get_submodule('0.model.stem.1')
list(l.parameters())
```
![](https://i.imgur.com/DwPQdOe.png)

## How does a neural net really work?
[How does a neural net really work?](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work)

```
def f(x): return 3*x**2 + 2*x + 1

plot_function(f, "$3x^2 + 2x + 1$")
```
![](https://i.imgur.com/w9v2ei8.png)

### more easier to create quadratic formula
```
def quad(a, b, c, x): return a*x**2 + b*x + c

quad(3, 2, 1, 1.5)
10.75
```
```
from functools import partial
def mk_quad(a,b,c): return partial(quad, a,b,c)

f = mk_quad(3,2,1)
f(1.5)
10.75
```

### data never match shape of function
```
from numpy.random import normal,seed,uniform
# 產生常態分布的隨機數
def noise(x, scale): return np.random.normal(scale=scale, size=x.shape)
def add_noise(x, mult, add): return x * (1+noise(x,mult)) + noise(x,add)

np.random.seed(42)

x = torch.linspace(-2, 2, steps=20)[:,None]
y = add_noise(f(x), 0.15, 1.5)

plt.scatter(x,y);
```
![](https://i.imgur.com/PGZ1CSD.png)

### find quadratic quation which matches this data
```
from ipywidgets import interact
@interact(a=1.1, b=1.1, c=1.1)
def plot_quad(a, b, c):
    plt.scatter(x,y)
    plot_function(mk_quad(a,b,c), ylim=(-3,12))
```
![](https://i.imgur.com/b7tpOhC.png)

### better or worse
```
def mae(preds, acts): return (torch.abs(preds-acts)).mean()
```

```
@interact(a=1.1, b=1.1, c=1.1)
def plot_quad(a, b, c):
    f = mk_quad(a,b,c)
    plt.scatter(x,y)
    loss = mae(f(x), y)
    plot_function(f, ylim=(-3,12), title=f"MSE: {loss:.2f}")
```
![](https://i.imgur.com/Py0X54B.png)

### Automating gradient descent
```
def quad_mae(params):
    f = mk_quad(*params)
    return mae(f(x), y)

quad_mae([1.1, 1.1, 1.1])
tensor(10.1370, dtype=torch.float64)
```

```
# rank 1 tensor
abc = torch.tensor([1.1,1.1,1.1])
abc.requires_grad_()
tensor([1.1000, 1.1000, 1.1000], requires_grad=True)
```

```
loss = quad_mae(abc)
loss
tensor(10.1370, dtype=torch.float64, grad_fn=<MeanBackward0>)
# grad_fn 代表可以幫我們計算輸出的gradient

# add grad attribute to abc parameter
loss.backward()

abc.grad
tensor([-11.3866,  -0.1188,  -3.8409])
# 當增加a，損失會減少，數字越大代表減少損失越多
```

```
# no_grad()代表這裡面的數據不用計算梯度，也不用backward()
# 0.01 為 learning rate
with torch.no_grad():
    abc -= abc.grad*0.01
    loss = quad_mae(abc)
    
print(f'loss={loss:.2f}')
loss=5.40
```

```
# optimization
for i in range(10):
    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad(): abc -= abc.grad*0.01 # 0.01 is learning rate
    print(f'step={i}; loss={loss:.2f}')

step=0; loss=5.40
step=1; loss=4.05
step=2; loss=2.83
step=3; loss=2.14
step=4; loss=2.23
step=5; loss=3.05
step=6; loss=4.31
step=7; loss=5.54
step=8; loss=6.30
step=9; loss=6.31
```

### How a neural network approximates any given function
```
# 線性整流函式在基於斜坡函式的基礎上有其他同樣被廣泛應用於深度學習的變種，譬如帶泄露線性整流(Leaky ReLU)，帶泄露隨機線性整流(Randomized Leaky ReLU)，以及噪聲線性整流(Noisy ReLU)
def rectified_linear(m,b,x):
    y = m*x+b
    return torch.clip(y, 0.)

# nstead of torch.clip(y, 0.), we can instead use F.relu(x)
# import torch.nn.functional as F
# def rectified_linear2(m,b,x): return F.relu(m*x+b)
plot_function(partial(rectified_linear, 1,1))
```
![](https://i.imgur.com/1s0vsOZ.png)

```
@interact(m=1.5, b=1.5)
def plot_relu(m, b):
    plot_function(partial(rectified_linear, m,b), ylim=(-1,4))
```
![](https://i.imgur.com/K6Efx92.png)


```
def double_relu(m1,b1,m2,b2,x):
    return rectified_linear(m1,b1,x) + rectified_linear(m2,b2,x)

@interact(m1=-1.5, b1=-1.5, m2=1.5, b2=1.5)
def plot_double_relu(m1, b1, m2, b2):
    plot_function(partial(double_relu, m1,b1,m2,b2), ylim=(-1,6))
```
![](https://i.imgur.com/GwpbSbz.png)

透過簡單的基礎(ReLU)，可以建立一個任意的、準確的、精確的模型
但是需要參數，我們可以使用梯度下降

### How to draw an owl
![](https://i.imgur.com/PajkYPR.png)

將ReLU相加，用梯度下降優化參數，輸入想要的輸入和輸出，就能畫出貓頭鷹

### try out all model and find the best performing one
可以用最快的方式嘗試大量外部數據，並清理數據
嘗試更好的架構會是最後一件事

非監督學習
=>機器學習的一種方法，沒有給定事先標記過的訓練範例，自動對輸入的資料進行分類或分群。

### Matrix Multiplication
[Matrix Multiplication](http://matrixmultiplication.xyz/)

![](https://i.imgur.com/knHpEVj.png)
![](https://i.imgur.com/R7QDsWt.png)

### Titanic - Machine Learning from Disaster
[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/)

[Excel for machine learning](https://docs.google.com/spreadsheets/d/13MnqG3usNh9ubeSItogk7YlV4WsQUZNCCHpBqOEH4pI/edit?usp=sharing)

### MNIST Basic
[MNIST Basic](https://www.kaggle.com/leogodone/3-and-7-model)

### Pixel Similarity

```
path = untar_data(URLs.MNIST_SAMPLE)
path.ls()
(#3) [Path('valid'),Path('labels.csv'),Path('train')]
(path/'train').ls()
(#2) [Path('train/7'),Path('train/3')]

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes
(#6131) [Path('train/3/10.png'),Path('train/3/10000.png'),Path('train/3/10011.png'),Path('train/3/10031.png'),Path('train/3/10034.png'),Path('train/3/10042.png'),Path('train/3/10052.png'),Path('train/3/1007.png'),Path('train/3/10074.png'),Path('train/3/10091.png')...]

im3_path = threes[1]
im3 = Image.open(im3_path)
im3
```
![](https://i.imgur.com/oNqYWeR.png)

```
#hide_output
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```
![](https://i.imgur.com/I5FQ4ON.png)

```
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors),type(seven_tensors)
(6131, 6265, list)

show_image(three_tensors[1]);
```
![](https://i.imgur.com/IiJYEIl.png)

```
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape
torch.Size([6131, 28, 28])
```

```
len(stacked_threes.shape)
3

stacked_threes.ndim
3
```

input：n x m 
mean(0)：對每列同位置的值做平均，output：m
mean(1)：對每列做平均，，output：n
```
mean3 = stacked_threes.mean(0)
show_image(mean3);
```
![](https://i.imgur.com/IXtMpUG.png)

```
mean7 = stacked_sevens.mean(0)
show_image(mean7);
```
![](https://i.imgur.com/b7vyvhy.png)

MSE對於錯誤的懲罰較大，可用來檢測離群值
如果認為只是單純資料誤差，就能使用MAE
```
a_3 = stacked_threes[1]

# L1 norm、MAE(Mean Absolute Error)
dist_3_abs = (a_3 - mean3).abs().mean()
# L2 norm、MSE(Mean Square Error)
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs,dist_3_sqr
(tensor(0.1114), tensor(0.2021))

dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs,dist_7_sqr
(tensor(0.1586), tensor(0.3021))
```

PyTorch 也有提供這2個loss function，可從torch.nn.functional找到，PyTorch建議 import as F(fastai 也是使用這個當預設名字)
```
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
(tensor(0.1586), tensor(0.3021))
```

```
data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)

# numpy
arr
array([[1, 2, 3],
       [4, 5, 6]])

# pytorch
tns
tensor([[1, 2, 3],
        [4, 5, 6]])
```

### Computing Metrics Using Broadcasting
```
valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape
(torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))
```

```
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)
tensor(0.1114)

valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
(tensor([0.1270, 0.1632, 0.1676,  ..., 0.1228, 0.1210, 0.1287]),
 torch.Size([1010]))
```

```
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)

is_3(a_3), is_3(a_3).float()
(tensor(True), tensor(1.))

is_3(valid_3_tens)
tensor([ True, False, False,  ...,  True,  True, False])

accuracy_3s =      is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
(tensor(0.9168), tensor(0.9854), tensor(0.9511))
```

### Stochastic Gradient Descent (SGD)
![](https://i.imgur.com/KIcAUS1.png)

```
def f(x): return x**2
plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red');
```
![](https://i.imgur.com/knCVmFE.png)

### Calculating Gradients
```
xt = tensor(3.).requires_grad_()
yt = f(xt)
yt
tensor(9., grad_fn=<PowBackward0>)

yt.backward()
xt.grad
tensor(6.)
```

### Stepping With a Learning Rate

```
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c

time = torch.arange(0,20).float(); time
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])

speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time,speed);
```
![](https://i.imgur.com/yiA4StD.png)


Step 1: Initialize the parameters
```
params = torch.randn(3).requires_grad_()
params
tensor([ 0.0103,  2.0304, -0.0076], requires_grad=True)
```
Step 2: Calculate the predictions
```
preds = f(time, params)
def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)

show_preds(preds)
```

![](https://i.imgur.com/0YWXIB4.png)

Step 3: Calculate the loss
```
loss = mse(preds, speed)
loss
tensor(695.4462, grad_fn=<MeanBackward0>)
```

Step 4: Calculate the gradients
```
loss.backward()
params.grad
tensor([684.8101,  81.1828,  -7.9379])
```

Step 5: Step the weights.
```
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None

preds = f(time,params)
mse(preds, speed)
tensor(692.0293, grad_fn=<MeanBackward0>)
```

```
def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds
```

Step 6: Repeat the process
```
for i in range(10): apply_step(params)
692.029296875
691.37060546875
691.2337646484375
691.1956787109375
691.1764526367188
691.1605224609375
691.1454467773438
691.13037109375
691.115478515625
691.1004638671875
```

Step 7: stop
觀察訓練組損失和驗證損失以及指標，才會決定何時停止

### The MNIST Loss Function
```
# -1 代表足夠大的容量能容納所有的資料
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)

train_x.shape,train_y.shape
(torch.Size([12396, 784]), torch.Size([12396, 1]))

dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y
(torch.Size([784]), tensor([1]))

# valid set
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

我們希望權重產生好一點的預測，損失函數可以產生好一點的損失，以該例來說，正確答案是3，分數就會高一些，反之，分數就會低一些

```
def mnist_loss(predictions, targets):
    # where(a, b, c) => (a) ? b : c
    return torch.where(targets==1, 1-predictions, predictions).mean()

# 有3張圖分別為3,7,3，有0.9的信心預測第1個是3，有0.4的信心預測第2個是7，有0.2的信心但錯誤預測第3個是7
trgts  = tensor([1,0,1])
prds   = tensor([0.9, 0.4, 0.2])
mnist_loss(prds,trgts)
tensor(0.4333)

# 將有0.8的信心預測第3個是3
prds   = tensor([0.9, 0.4, 0.8])
mnist_loss(prds,trgts)
tensor(0.2333)
```

```
# 不管傳入正或副數，都只會回傳0~1
def sigmoid(x): return 1/(1+torch.exp(-x))

plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)
```
![](https://i.imgur.com/ehTrLH6.png)

有準確度，為什麼還要定義損失，差別在於前者協助人類理解，損失為了自動化學習，損失必須是個有意義的導數函式，不能有很大的平坦區或是很大的跳躍，所以我們才設計出對信心度有微小改變就能做出反應的損失函數
有時無法反應出試著達到的目標，而是在真正目標和可用梯度優化函數中間取得妥協的方案

### SGD and Mini-Batches
為整個資料組計算損失或是為一個資料項目算損失，這2種做法都不理想，前者需要很多時間，後者使用資訊太少，算出不精確且不穩定的梯度

小批次(Mini-Batches)，一次計算一些資料項目的平均損失，小批次的資料項目數量稱為批次大小(batch size)

在訓練期變動內容，可以得到更好的類推能力，通常不會在每個epoch單純按照順序放入資料組，會隨機洗亂，再建立小批次

```
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)
[tensor([ 0,  7,  4,  5, 11]),
 tensor([ 9,  3,  8, 14,  6]),
 tensor([12,  2,  1, 10, 13])]
```
但不能用上述任意python集合來訓練模型，必須包含輸入和目標
```
ds = L(enumerate(string.ascii_lowercase))
ds
(#26) [(0, 'a'),(1, 'b'),(2, 'c'),(3, 'd'),(4, 'e'),(5, 'f'),(6, 'g'),(7, 'h'),(8, 'i'),(9, 'j')...]

dl = DataLoader(ds, batch_size=6, shuffle=True)
list(dl)
[(tensor([ 6, 14, 12, 15, 24, 11]), ('g', 'o', 'm', 'p', 'y', 'l')),
 (tensor([ 0, 16,  2, 18, 25, 21]), ('a', 'q', 'c', 's', 'z', 'v')),
 (tensor([ 8,  7, 19, 23,  1,  9]), ('i', 'h', 't', 'x', 'b', 'j')),
 (tensor([ 4, 13, 10,  5,  3, 17]), ('e', 'n', 'k', 'f', 'd', 'r')),
 (tensor([22, 20]), ('w', 'u'))]
```

### Putting It All Together
```
# 可以使用pytorch的nn.Linear模組
linear_model = nn.Linear(28*28,1)

# 可以使用parameters()知道哪些參數可以訓練
w,b = linear_model.parameters()
w.shape,b.shape
(torch.Size([1, 784]), torch.Size([1]))

# fastai有SGD類別，預設情形做的事情會和我們自己寫SGD的函數一樣
opt = SGD(linear_model.parameters(), lr)
```

```
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

dls = DataLoaders(dl, valid_dl)
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(10, lr=lr)
```
![](https://i.imgur.com/rAUTqDa.png)

### Adding a Nonlinearity
```
# 基本神經網路定義
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```

res.max(tensor(0.0))被稱為整流線性單位函數(rectified linear unit, ReLU)，功能是將所有負數都換成0，pytorch也有提供F.relu
```
plot_function(F.relu)
```
![](https://i.imgur.com/7suDh7G.png)

藉由更多線性層，能讓模型做更多計算，單純把n個線性函數加在一起，可以看成一個參數不一樣的線性層，如果在線性層之間放入非線性函數，線性層就會彼此脫鉤

```
# nn.Sequential是模組，依次呼叫列出來的層
# nn.ReLU()這層稱為非線性函數(nonlinearity、activation function)
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)
```

```
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
                
learn.fit(40, 0.1)
```
![](https://i.imgur.com/ZlEssdm.png)

```
# 訓練程序都記錄在 learn.recorder
plt.plot(L(learn.recorder.values).itemgot(2));
```
![](https://i.imgur.com/wFMVhbk.png)

A function that can solve any problem to any level of accuracy (the neural network) given the correct set of parameters

A way to find the best set of parameters for any function (stochastic gradient descent)

### Going Deeper
```
dls = ImageDataLoaders.from_folder(path)
learn = vision_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
```
![](https://i.imgur.com/7QmNisv.png)

從前面知道，一個非線性函數和兩個線性層足以近似任何函數，為何要使用更深的模型，理由出在效能，使用更深的模型就不需要那麼多參數，使用更小的矩陣和更多層可以產生比使用更大的矩陣和更少層更好的結果，但模型越深，參數也越南優化