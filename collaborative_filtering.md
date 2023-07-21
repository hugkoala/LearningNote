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