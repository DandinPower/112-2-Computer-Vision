## 112-2 Computer Vision HW1 Report

智能所 312581029 廖永誠

## Questions:

1. Discuss the results of blurred images and detected edges between different kernel sizes of Gaussian filter.

    - 在 blurred 的效果上，使用不同的 kernel size 會有不同的效果，當 kernel size 越大時，模糊的效果越明顯，因為 kernel size 越大，代表平滑的程度越高，因此模糊的效果也越明顯。
    - 在 detected edges 的部分，kernel size 越大，代表平滑的程度越高，因此 detected edges 的效果會越不明顯，因為平滑的效果會將邊緣的部分模糊掉，因此 detected edges 的效果會越不明顯。但同樣的抗雜訊能力會提高。

2. Discuss the difference between 3x3 and 30x30 window sizes of structure tensor.

    - 可以觀察到當 window size 較小的時候，能夠偵測到較細節的 Gradient 變化，因此能夠偵測到較細節的角點，但是也會因為 window size 較小，導致抗雜訊的能力較差，導致偵測到的角點較多。
    - 反之，當 window size 較大的時候，能夠偵測到較大範圍的 Gradient 變化，因此能夠偵測到較大範圍的角點，但是也會因為 window size 較大，導致偵測到的角點較少，但是抗雜訊的能力較好。
    - 結果會因為 window size 的不同而有所不同。要根據圖片的特性來選擇適合的 window size。
 
3. Discuss the effect of non-maximal suppression.

    - non-maximal suppression 會在 corner response matrix 中，以一定範圍做為 neighbor，挑選出最大的值且大於 threshold 的值，將其視為 corner point。
    - 如果不做 non-maximal suppression，則會將所有的 corner point 都視為 corner point，導致 corner point 的數量過多，因此 non-maximal suppression 可以過濾掉一些不重要的 corner point。
    - 在觀察開啟 non-maximal suppression 與否的結果時，可以發現開啟 non-maximal suppression 的結果，corner point 的數量會較少，但是 corner point 的品質會較好。

4. Discuss the results of rotated and scaled image. Is Harris detector rotation invariant or scale-invariant? Explain the reason.

    - 根據觀察及理論分析，Harris detector 是 rotation invariant，但是不是 scale-invariant。這是因為 Harris detector 是透過計算 corner response matrix 來找出 corner point，而 corner response matrix 是透過計算 gradient 的變化來計算的，因此對於 rotation 會有不錯的表現，但是對於 scale 的變化，則會有一定的影響，因為不論是 gaussian filter 或是 structure tensor 都會受到 scale 的影響，因為期會有固定的 window size，但是 rotation 的變化則不會受到影響，因為 rotation 並不會影響到 gradient 的計算。
    - 不過從最終的結果來看，即使是 scale 的變化，Harris detector 也能夠偵測到 corner point。