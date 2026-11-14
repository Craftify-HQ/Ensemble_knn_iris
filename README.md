# Ensemble Learning : KNN vs Bagging vs Boosting (Iris Dataset)

เปรียบเทียบประสิทธิภาพของโมเดลเดี่ยว **K-Nearest Neighbors (KNN)** กับ Ensemble 2 แบบ คือ **Bagging** และ **Boosting**
ในการจำแนกสายพันธุ์ดอก Iris โดยวัดทั้ง **ความแม่นยำ**, **เวลาเทรน** และ **เวลาทำนาย (inference)**

## โจทย์
- เลือก algorithm สำหรับ Classification 1 ตัว (ใช้ library ได้) → **KNN** จาก scikit-learn
- เขียน **Bagging** เองด้วย Python built-in + numpy แบบ **Parallel Computing** (`concurrent.futures.ThreadPoolExecutor`)
- เขียน **Boosting** (AdaBoost แบบ SAMME) เองแบบ Sequential
- ใช้ Iris Dataset แบ่ง **Train 80% / Test 20%** แล้วเปรียบเทียบทั้ง 3 แบบ

## ไฟล์ใน Repository
| ไฟล์ | รายละเอียด |
|---|---|
| `ensemble_knn_iris.ipynb` | โค้ดหลักทั้งหมด พร้อมคำอธิบายภาษาไทย ผลลัพธ์ และกราฟ |
| `iris.csv` | ข้อมูล Iris 150 ตัวอย่าง (4 features + species) ชุดเดียวกับที่ใช้ใน notebook |

## Dataset
- **Iris Dataset** (Fisher, 1936) โหลดจาก `sklearn.datasets.load_iris`
- 150 ตัวอย่าง, 4 features (ความยาว/กว้างของกลีบเลี้ยงและกลีบดอก), 3 สายพันธุ์ (setosa, versicolor, virginica) สายพันธุ์ละ 50
- แบ่ง Train 120 ตัวอย่าง / Test 30 ตัวอย่าง (`stratify=y`, `random_state=42`)

## วิธีการใช้งาน

### 1. ติดตั้ง
ต้องมี **Python 3.9 ขึ้นไป** แล้วติดตั้ง library:
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

### 2. ดาวน์โหลดโปรเจกต์
```bash
git clone https://github.com/Craftify-HQ/Ensemble_knn_iris.git
cd Ensemble_knn_iris
```
หรือกดปุ่ม **Code → Download ZIP** บนหน้า GitHub แล้วแตกไฟล์

### 3. เปิดและรัน Notebook
เลือกวิธีใดวิธีหนึ่ง:

- **VS Code** : เปิดไฟล์ `ensemble_knn_iris.ipynb` → เลือก Python kernel มุมขวาบน → กด **Run All**
- **Jupyter Notebook** :
  ```bash
  jupyter notebook ensemble_knn_iris.ipynb
  ```
  แล้วเลือกเมนู **Kernel → Restart & Run All**
- **Google Colab** : ไปที่ [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → เลือกไฟล์ `.ipynb` → กด **Runtime → Run all** (Colab มี library ครบอยู่แล้ว ไม่ต้องติดตั้ง)

> รันทั้ง notebook ใช้เวลาไม่ถึง 1 นาที ข้อมูลโหลดจาก scikit-learn โดยตรง จึงไม่ต้องมีไฟล์ `iris.csv` ก็รันได้

### 4. (ทางเลือก) อ่านข้อมูลจาก `iris.csv` แทน
แก้ cell ในหัวข้อ **"2. Import Data"** เป็น:
```python
df = pd.read_csv("iris.csv")
X = df[["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]].to_numpy()
class_names = np.array(sorted(df["species"].unique()))
y = np.searchsorted(class_names, df["species"])
```

### 5. ปรับค่าที่ทดลองได้
| ค่า | ตำแหน่ง | ความหมาย |
|---|---|---|
| `K = 5` | หัวข้อ 4 | จำนวนเพื่อนบ้านของ KNN |
| `n_estimators=50` | หัวข้อ 5 และ 6 | จำนวนโมเดลย่อยใน Bagging / Boosting |
| `n_workers=4` | class `MyBagging` | จำนวน thread ที่ใช้ทำงานขนาน |
| `test_size=0.2` | หัวข้อ 2 | สัดส่วนข้อมูล Test |

## ขั้นตอนใน Notebook
1. **Import Library**
2. **Import Data** และแบ่ง Train/Test 80/20 พร้อมปรับสเกลด้วย `StandardScaler`
3. **ฟังก์ชัน `evaluate`** วัด accuracy และจับเวลาเทรน/ทำนายด้วยวิธีเดียวกันทุกโมเดล
4. **KNN เดี่ยว (Baseline)**
5. **Bagging (`MyBagging`)**
   - สุ่ม Bootstrap ด้วย numpy
   - เทรนและทำนาย 50 โมเดลแบบขนานด้วย `ThreadPoolExecutor`
   - รวมผลด้วย Majority Vote
6. **Boosting (`MyAdaBoost`)**
   - เริ่มให้ทุกตัวอย่างมีน้ำหนักเท่ากัน
   - แต่ละรอบสุ่มข้อมูลตามน้ำหนัก → เทรน KNN → เพิ่มน้ำหนักตัวอย่างที่ทายผิด
   - รวมผลด้วย Weighted Vote (ค่า alpha)
7. **เปรียบเทียบผล** ตาราง, กราฟ accuracy / เวลา และ Confusion Matrix
8. **ทดลองเพิ่ม** จำนวนโมเดลย่อย (1–100) มีผลต่อ accuracy และเวลาอย่างไร
9. **สรุปผล**

## ผลลัพธ์
| Model | Train Acc | Test Acc | ทายถูก (Test) |
|---|---|---|---|
| KNN (single) | 97.50% | 93.33% | 28 / 30 |
| Bagging (KNN x50) | 97.50% | 93.33% | 28 / 30 |
| Boosting (KNN x50) | 100.00% | **96.67%** | 29 / 30 |

> เวลาเทรน/ทำนายจะต่างกันเล็กน้อยในแต่ละเครื่องและแต่ละครั้งที่รัน

**สรุป**
- Boosting แม่นยำที่สุด แต่ต่างจากแบบอื่นเพียง 1 ตัวอย่าง และใช้เวลาเทรนมากที่สุดเพราะทำงานแบบ Sequential
- Bagging ได้ความแม่นยำเท่ากับ KNN เดี่ยว เพราะ KNN เป็นโมเดลที่มีความแปรปรวนต่ำ การโหวตจึงไม่ช่วยเพิ่มความแม่นยำ
- Bagging เทรนเร็วกว่า Boosting เพราะทำงานแบบขนานได้ แต่บนข้อมูลขนาดเล็ก overhead ของ thread ทำให้ช้ากว่าโมเดลเดี่ยวมาก
- สำหรับข้อมูลที่เล็กและแยกคลาสง่ายอย่าง Iris โมเดลเดี่ยวคุ้มค่าที่สุด Ensemble จะคุ้มค่าเมื่อข้อมูลซับซ้อนหรือมี noise

## Reference
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179–188.
- [UCI Machine Learning Repository : Iris](https://archive.ics.uci.edu/dataset/53/iris)
- [scikit-learn : load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
