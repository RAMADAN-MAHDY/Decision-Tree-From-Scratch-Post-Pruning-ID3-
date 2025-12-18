# Decision Tree From Scratch + Post-Pruning (ID3)

## وصف عام
هذا المشروع ينفذ خوارزمية شجرة القرار (ID3) من الصفر باستخدام بايثون فقط (بدون مكتبات تعلم آلي مثل scikit-learn)، ويشمل أيضًا تقليم الشجرة (post-pruning) لتحسين الأداء.

---

## شرح الكود سطر بسطر

### 1. استيراد المكتبات
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
- **pandas**: لتحميل ومعالجة البيانات.
- **numpy**: للعمليات العددية والمصفوفات.
- **sklearn.model_selection.train_test_split**: لتقسيم البيانات إلى تدريب/اختبار/تحقق.
- **sklearn.metrics**: لحساب دقة النموذج وطباعه تقارير التصنيف.

---

### 2. تحميل البيانات
```python
df = pd.read_csv("academic_performance_1000.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())
```
- تحميل ملف البيانات.
- طباعة شكل الجدول (عدد الصفوف والأعمدة).
- طباعة أسماء الأعمدة.
- التحقق من وجود قيم مفقودة.

---

### 3. حساب الإنتروبي (Entropy)
```python
def entropy(y):
    if len(y) == 0:
        return 0.0
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    ent = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return ent
```
- **entropy(y)**: تحسب إنتروبي شانون لمصفوفة التصنيفات `y`.
- إذا كانت `y` فارغة تعيد صفر.
- تحسب احتمالية كل تصنيف.
- تطبق معادلة الإنتروبي.

---

### 4. حساب كسب المعلومات (Information Gain)
```python
def information_gain(X_col, y, split_val):
    left_mask = X_col <= split_val
    right_mask = X_col > split_val
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return 0
    ent_before = entropy(y)
    y_left = y[left_mask]
    y_right = y[right_mask]
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = len(y)
    ent_after = (n_left / n_total) * entropy(y_left) + (n_right / n_total) * entropy(y_right)
    ig = ent_before - ent_after
    return ig
```
- **information_gain**: تحسب كسب المعلومات عند تقسيم العمود `X_col` عند قيمة معينة.
- تقسم البيانات إلى قسمين (يسار ويمين).
- تحسب الإنتروبي قبل وبعد التقسيم.
- تعيد الفرق (كسب المعلومات).

---

### 5. إيجاد أفضل تقسيم (Best Split)
```python
def best_split(X, y):
    best_gain = -1
    best_feature = None
    best_value = None
    n_features = X.shape[1]
    for feature_idx in range(n_features):
        X_col = X[:, feature_idx]
        thresholds = np.unique(X_col)
        for threshold in thresholds:
            ig = information_gain(X_col, y, threshold)
            if ig > best_gain:
                best_gain = ig
                best_feature = feature_idx
                best_value = threshold
    return best_feature, best_value, best_gain
```
- تبحث عن أفضل ميزة وأفضل قيمة تقسيم تعطي أعلى كسب معلومات.
- تجرب كل ميزة وكل قيمة ممكنة.

---

### 6. تعريف شجرة القرار (DecisionTreeScratch)
```python
class DecisionTreeScratch:
    class Node:
        def __init__(self, feature=None, value=None, left=None, right=None, label=None):
            self.feature = feature
            self.value = value
            self.left = left
            self.right = right
            self.label = label
    def __init__(self, max_depth=6, min_samples=12):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
    def build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        if n_classes == 1:
            return self.Node(label=y[0])
        if depth >= self.max_depth or n_samples < self.min_samples:
            unique, counts = np.unique(y, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            return self.Node(label=majority_label)
        best_feature, best_value, best_gain = best_split(X, y)
        if best_feature is None or best_gain == 0:
            unique, counts = np.unique(y, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            return self.Node(label=majority_label)
        left_mask = X[:, best_feature] <= best_value
        right_mask = X[:, best_feature] > best_value
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        return self.Node(feature=best_feature, value=best_value, left=left_subtree, right=right_subtree)
    def _predict_one(self, node, x):
        if node.label is not None:
            return node.label
        if x[node.feature] <= node.value:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)
    def predict(self, X):
        return np.array([self._predict_one(self.tree, x) for x in X])
```
- **Node**: يمثل عقدة في الشجرة (إما تقسيم أو ورقة).
- **build_tree**: يبني الشجرة بشكل متكرر (recursively) حسب شروط التوقف.
- **_predict_one**: يتنبأ بتصنيف عينة واحدة.
- **predict**: يتنبأ لمجموعة عينات.

---

### 7. تقليم الشجرة (Post-Pruning)
```python
def prune_tree(node, X_val, y_val, model):
    if node.label is not None:
        return node
    node.left = prune_tree(node.left, X_val, y_val, model)
    node.right = prune_tree(node.right, X_val, y_val, model)
    y_pred_before = model.predict(X_val)
    accuracy_before = accuracy_score(y_val, y_pred_before)
    unique, counts = np.unique(y_val, return_counts=True)
    majority_label = unique[np.argmax(counts)]
    original_feature = node.feature
    original_value = node.value
    original_left = node.left
    original_right = node.right
    node.feature = None
    node.value = None
    node.left = None
    node.right = None
    node.label = majority_label
    y_pred_after = model.predict(X_val)
    accuracy_after = accuracy_score(y_val, y_pred_after)
    if accuracy_after >= accuracy_before:
        return node
    else:
        node.feature = original_feature
        node.value = original_value
        node.left = original_left
        node.right = original_right
        node.label = None
        return node
```
- **prune_tree**: يحاول تقليم كل عقدة غير ورقية.
- إذا زادت الدقة أو لم تتغير بعد التقليم، يحتفظ بالتقليم.
- إذا قلت الدقة، يعيد العقدة كما كانت.

---

### 8. تقسيم البيانات
```python
X = df.drop("Pass", axis=1).values
y = df["Pass"].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```
- تقسيم البيانات إلى تدريب (60%)، تحقق (20%)، واختبار (20%).

---

### 9. تدريب وتقييم الشجرة
```python
model = DecisionTreeScratch(max_depth=12, min_samples=5)
model.tree = model.build_tree(X_train, y_train)
y_pred_unpruned = model.predict(X_test)
accuracy_unpruned = accuracy_score(y_test, y_pred_unpruned)
print("Unpruned Tree Test Accuracy:", accuracy_unpruned)
print(classification_report(y_test, y_pred_unpruned))
model.tree = prune_tree(model.tree, X_val, y_val, model)
y_pred_pruned = model.predict(X_test)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print("Pruned Tree Test Accuracy:", accuracy_pruned)
print(classification_report(y_test, y_pred_pruned))
```
- تدريب الشجرة بدون تقليم.
- حساب الدقة والتقارير.
- تقليم الشجرة باستخدام مجموعة التحقق.
- إعادة التقييم بعد التقليم.

---

### 10. مناقشة النتائج
- تقليم الشجرة يقلل التعقيد ويحسن التعميم.
- قد تتغير بعض المقاييس لكل فئة، لكن الهدف الأساسي هو تقليل الإفراط في التخصيص.

---

## ملخص
- الكود يطبق شجرة قرار ID3 من الصفر.
- يشمل تقليم بعدي (post-pruning) لتحسين الأداء.
- كل خطوة مشروحة بالتفصيل أعلاه.

---

**بالتوفيق في المذاكرة!**