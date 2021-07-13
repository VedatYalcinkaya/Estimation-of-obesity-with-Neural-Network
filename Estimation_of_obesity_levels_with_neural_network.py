import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gender(answer):
    if answer == "Female" or (answer == "Kadın" or answer == "kadın"):
        return 1
    elif answer == "Male" or (answer == "Erkek" or answer == "erkek"):
        return 2


def yesNo(answer):
    if answer == "yes" or (answer == "Evet" or answer == "evet"):
        return 1
    elif answer == "no" or (answer == "Hayır" or answer == "hayır"):
        return 0


def ordinal(answer):
    if answer == "no" or (answer == "Hayır" or answer == "hayır"):
        return 0
    elif answer == "Sometimes" or (answer == "Bazen" or answer == "bazen"):
        return 1
    elif answer == "Frequently" or (answer == "Sık sık" or answer == "sık sık"):
        return 2
    elif answer == "Always" or (answer == "Her zaman" or answer == "her zaman"):
        return 4


def transportation(answer):
    if answer == "Automobile" or (answer == "Otomobil" or answer == "otomobil"):
        return 1
    elif answer == "Motorbike" or (answer == "Motor" or answer == "motor"):
        return 2
    elif answer == "Public_Transportation" or (answer == "Toplu taşıma" or answer == "toplu taşıma"):
        return 3
    elif answer == "Walking" or (answer == "Yürüyüş" or answer == "yürüyüş"):
        return 4
    elif answer == "Bike" or (answer == "Bisiklet" or answer == "bisiklet"):
        return 5


def obesityLevels(level):
    if level == "Insufficient_Weight":
        return 0.125
    elif level == "Normal_Weight":
        return 0.25
    elif level == "Overweight_Level_I":
        return 0.375
    elif level == "Overweight_Level_II":
        return 0.5
    elif level == "Obesity_Type_I":
        return 0.625
    elif level == "Obesity_Type_II":
        return 0.75
    elif level == "Obesity_Type_III":
        return 0.875


def normalizasyon(inputs):
    attrSize = inputs.shape[1] - 1  # Column Sayısı
    for i in range(attrSize):
        max = inputs[:, i].max()  # Column max değeri
        min = inputs[:, i].min()  # Column min değeri
        for j in range(len(inputs)):
            inputs[j, i] = (inputs[j, i] - min) / (
                        max - min)  # 𝑋𝑦𝑒𝑛𝑖=((𝑋−𝑋𝑚𝑖𝑛)/(𝑋𝑚𝑎𝑥−𝑋𝑚𝑖𝑛))(𝑏−𝑎)+𝑎
    return inputs


df = pd.read_csv('ObesityDataSet.csv')  # DataSeti Okuma

# Kategorik verileri sayısal hale getirmek için fonksiyonları kullanma
df["NObeyesdad"] = df["NObeyesdad"].apply(obesityLevels)
df["Gender"] = df["Gender"].apply(gender)
df["family_history_with_overweight"] = df["family_history_with_overweight"].apply(yesNo)
df["FAVC"] = df["FAVC"].apply(yesNo)
df["SMOKE"] = df["SMOKE"].apply(yesNo)
df["SCC"] = df["SCC"].apply(yesNo)
df["CAEC"] = df["CAEC"].apply(ordinal)
df["CALC"] = df["CALC"].apply(ordinal)
df["MTRANS"] = df["MTRANS"].apply(transportation)

# Normalizasyon uygulanması
df_matris = df.to_numpy()
df_normalize = normalizasyon(df_matris)
labels = ["Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE",
          "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS", "NObeyesdad"]
df_normalize = pd.DataFrame(df_normalize, columns=labels)

# Data seti rastgele sıralama
df_random = df_normalize
df_random["Random"] = np.random.rand(2111)
df_random.sort_values("Random", inplace=True)
df_random.reset_index(drop=True, inplace=True)
df_random.drop(["Random"], axis=1, inplace=True)

# Eğitim ve Test Sayısı
df_train = df_random.head(1500)
df_test = df_random.tail(611)
df_test.reset_index(drop=True, inplace=True)

# Eğitim Verisi
data_train = df_train.to_numpy()
data_train_op = data_train[:, -1]
data_train_op = np.array([data_train_op])
data_train_op = data_train_op.T
data_train_inputs = data_train[:, :16]
data_train_inputs = np.array(data_train_inputs)

# Test Verisi
data_test = df_test.to_numpy()
data_test_op = data_test[:, -1]
data_test_op = np.array([data_test_op])
data_test_op = data_test_op.T
data_test_inputs = data_test[:, :16]
data_test_inputs = np.array(data_test_inputs)


class NeuralNetwork():
    def __init__(self):
        self.inputSize = 16
        self.outputSize = 1
        self.hiddenSize = 8
        self.lr = 0.8
        self.nu = 0.5

        # giriş katmanı => gizli katman
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.deltaW1 = np.zeros((self.inputSize, self.hiddenSize))

        # gizli katman => çıkış katmanı
        self.W = np.random.randn(self.hiddenSize, self.outputSize)
        self.deltaW = np.zeros((self.hiddenSize, self.outputSize))

        self.bias1 = np.random.randn(1, self.hiddenSize)
        self.bias = np.random.randn(1, self.outputSize)

        self.deltaBias1 = np.zeros((1, self.hiddenSize))
        self.deltaBias = np.zeros((1, self.outputSize))

    def feedForward(self, inputs):
        self.Op1 = np.dot(inputs, self.W1) + self.bias1
        self.Op1 = self.sigmoid(self.Op1)
        self.Op = np.dot(self.Op1, self.W) + self.bias
        output = self.sigmoid(self.Op)
        return output

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def train(self, inputs, realOutput):
        inputs = np.array([inputs])
        realOutput = np.array([realOutput])
        output = self.feedForward(inputs)
        self.backProp(inputs, realOutput, output)

    def backProp(self, inputs, realOutput, output):

        self.output_error = (realOutput - output) * output * (1 - output)
        self.deltaW = (self.output_error * self.Op1 * self.nu).T + (self.lr * self.deltaW)
        self.W += self.deltaW  # hidden ->output weights

        self.hidden_error = np.dot(self.output_error, self.W.T) * self.Op1 * (1 - self.Op1)
        self.deltaW1 = self.nu * np.dot(inputs.T, self.hidden_error) * self.nu + self.lr * self.deltaW1
        self.W1 += self.deltaW1  # input -> hidden weights

        self.deltaBias = self.nu * self.output_error + self.deltaBias * self.lr
        self.bias += self.deltaBias

        self.deltaBias1 = self.hidden_error * self.nu + self.deltaBias1 * self.lr
        self.bias1 += self.deltaBias1

    def accuracy(self, inputs, outputs):
        num_correct = 0.0
        preds = []
        for i in range(len(inputs)):
            pred = self.predict(inputs[i], Break=True)
            preds.append(pred)
            # print(i,".örnek tahmin :",preds[i],outputs[i])
            # if (pred <= 0.625 and outputs[i] <= 0.625) or (pred > 0.625 and outputs[i] > 0.625): num_correct += 1
            if pred == outputs[i]: num_correct += 1
        return num_correct / float(len(inputs))

    def predict(self, inputs, Break=False):
        Op1 = np.dot(inputs, self.W1) + self.bias1
        Op1 = self.sigmoid(Op1)
        Op = np.dot(Op1, self.W) + self.bias
        output = self.sigmoid(Op)
        if Break:
            if output <= 0.16:
                return 0.125
            elif output <= 0.32:
                return 0.25
            elif output <= 0.43:
                return 0.375
            elif output <= 0.53:
                return 0.5
            elif output <= 0.67:
                return 0.625
            elif output <= 0.81:
                return 0.75
            else:
                return 0.875
        return output


class Estimate():

    def __init__(self):
        self.trainNN()

    def trainNN(self):
        self.NN = NeuralNetwork()
        try:
            self.epoch = int(input("Epoch sayısı : "))
        except:
            print("Lütfen sayısal değer girin ")
            self.epoch = int(input("Epoch sayısı 1: "))
        self.network(self.NN, self.epoch)
        return self.NN

    def network(self, NN, epoch):
        self.loss_list = []
        self.epoch_list = []
        self.accuracy_list = []
        self.train_accuracy_list = []

        for i in range(self.epoch):
            for j in range(len(data_train_inputs)):
                NN.train(data_train_inputs[j], data_train_op[j])

            self.accuracy_list.append(NN.accuracy(data_test_inputs, data_test_op))
            self.train_accuracy_list.append(NN.accuracy(data_train_inputs, data_train_op))
            self.epoch_list.append(i)
            self.loss_list.append(np.mean(np.square(data_train_op - NN.predict(data_train_inputs))))
            if i % 1 == 0:
                print(self.epoch_list[i] + 1, ". Epoch loss : ",
                      str(np.mean(np.square(data_train_op - NN.predict(data_train_inputs)))))
                print(self.epoch_list[i] + 1, ". Epoch Train Accuruacy : ",
                      str(NN.accuracy(data_train_inputs, data_train_op)))
                print(self.epoch_list[i] + 1, ". Epoch Accuruacy : ", str(NN.accuracy(data_test_inputs, data_test_op)),
                      "\n")
            if self.accuracy_list[i] >= 1:
                break

    def graphic(self, loss_list, epoch_list, accuracy_list,train_accuracy_list):

        # Loss Grafiği
        fig = plt.figure(figsize=(8, 6))
        axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes2 = fig.add_axes([0.4, 0.25, 0.4, 0.4])
        axes1.set_title("Accuracy")
        axes1.plot(epoch_list, accuracy_list, label="Test Accuracy", color="#992100", ls="-.")
        axes1.plot(epoch_list, train_accuracy_list, label="Train Accuracy", color="blue", ls="dotted")
        axes1.set_xlabel("Epoch")
        axes1.set_ylabel("Accuracy")
        axes2.set_title("Loss")
        axes2.plot(epoch_list, loss_list,label="Loss",color="red", ls="--")
        axes1.legend()
        plt.show()


    def getNumeric(self, personAnswers):
        personAnswers[0] = (gender(personAnswers[0]))
        personAnswers[4] = (yesNo(personAnswers[4]))
        personAnswers[5] = (yesNo(personAnswers[5]))
        personAnswers[9] = (yesNo(personAnswers[9]))
        personAnswers[11] = (yesNo(personAnswers[11]))
        personAnswers[8] = (ordinal(personAnswers[8]))
        personAnswers[14] = (ordinal(personAnswers[14]))
        personAnswers[15] = (transportation(personAnswers[15]))
        for i in range(len(personAnswers)):
            personAnswers[i] = float(personAnswers[i])
        person_inputs = np.array(personAnswers)
        return person_inputs

    def estimateNormalizasyon(self, dataset_inputs, person_inputs):
        attrSize = person_inputs.shape[0]  # Column Sayısı
        for i in range(attrSize):
            max = dataset_inputs[:, i].max()  # Dataset Column max değeri
            min = dataset_inputs[:, i].min()  # Dataset Column min değeri
            person_inputs[i] = float(person_inputs[i] - min) / (
                        max - min)  # 𝑋𝑦𝑒𝑛𝑖=((𝑋−𝑋𝑚𝑖𝑛)/(𝑋𝑚𝑎𝑥−𝑋𝑚𝑖𝑛))(𝑏−𝑎)+𝑎
        return person_inputs

    def result(self, person_op):
        if person_op == 0.125:
            return "Zayıf"
        elif person_op == 0.25:
            return "Normal"
        elif person_op == 0.375:
            return "Fazla kilolu level I"
        elif person_op == 0.5:
            return "Fazla kilolu level II"
        elif person_op == 0.625:
            return "Obezite level I"
        elif person_op == 0.75:
            return "Obezite level II"
        elif person_op == 0.875:
            return "Obezite level III"

    def estimation(self):
        print("Tahminin yapılması için 16 adet soruya cevap vermeniz lazım")
        self.gender = input("\nCinsiyet (Erkek ya da Kadın) :")
        self.age = input("\nYaş :")
        self.height = input("\nBoy metre cinsinden örnek (1.75):")
        self.weight = input("\nKilo kg cinsinden örnek (75):")
        self.family_history_with_overweight = input("\nAile geçmişinde fazla kilolu var mı?\n (Evet ya da hayır) : ")
        self.favc = input("\nÇoğunlukla yüksek kalorili yiyecekler mi tüketiyorsunuz?\n (Evet ya da Hayır) : ")
        self.fcvc = input("\nYemeklerinizde genellikle sebze yer misiniz?\n (1-3 öğüne göre) : ")
        self.ncp = input("\nGünde kaç ana öğün yiyorsunuz?\n (1-4) : ")
        self.caec = input("\nÖğün aralarında yiyecek tüketiyor musunuz?\n(Hayır, Bazen, Sık sık, Her zaman) : ")
        self.smoke = input("\nSigara kullanıyor musunuz?\n(Evet yada Hayır) : ")
        self.ch2o = input("\nGünlük ne kadar su içiyorsunuz?\n (1-3) : ")
        self.scc = input("\nGünlük Tükettiğiniz kalorileri izliyor musunuz?\n (Evet yada Hayır) : ")
        self.faf = input("\nNe sıklıkla fiziksel aktivite yapıyorsunuz?(0-4 arası puanlayın) : ")
        self.tue = input("\nCep telefonu, televizyon, bilgisayar vb. cihazları ne kadar süre kullanıyorsunuz?\n0-2 "
                         "saat -> 0\n3-5 saat -> 1\n5 saatten çok -> 2 : ")
        self.calc = input("\nNe sıklıkla alkol içiyorsunuz?\n(hayır, bazen, sık sık, her zaman) :")
        self.mtrans = input("\nGenellikle hangi ulaşım aracını kullanıyorsunuz?\n(Otomobil, Motor, Toplu taşıma, "
                            "Yürüyüş, Bisiklet) :")
        self.personAnswers = [self.gender, self.age, self.height, self.weight, self.family_history_with_overweight,
                              self.favc, self.fcvc, self.ncp, self.caec, self.smoke, self.ch2o, self.scc, self.faf,
                              self.tue, self.calc, self.mtrans]
        self.person_inputs = self.getNumeric(self.personAnswers)
        self.person_inputs = self.estimateNormalizasyon(df.to_numpy(), self.person_inputs)
        self.person_op = self.NN.predict(self.person_inputs, Break=True)
        self.person_op = self.result(self.person_op)
        print("\nObezite seviyeniz :", self.person_op + "\n")


print("""
    ******** Yapay sinir ağı ile Obezite Levelini belirleme programı ********

    1. Yapay sinir ağını eğit 

    2. Eğitilmiş yapay sinir ağının accuracy ve loss grafiklerini al

    3. Eğitilmiş yapay sinir ağını kullanarak obezite levelini belirle

    Çıkmak için 'q' ya basın

    *************************************************************************
    """)
while True:

    choice = input("İşlemi Seçiniz :")

    if choice == "q":
        print("Program Sonlandırılıyor....")
        break
    elif choice == "1":
        estimate = Estimate()
    elif choice == "2":
        try:
            estimate.graphic(estimate.loss_list, estimate.epoch_list, estimate.accuracy_list, estimate.train_accuracy_list)
        except:
            print("İlk yapay sinir ağını eğitmelisiniz")
    elif choice == "3":
        try:
            estimate.estimation()
        except:
            print("İlk yapay sinir ağını eğitmelisiniz")
    else:
        print("Geçersiz İşlem....")
