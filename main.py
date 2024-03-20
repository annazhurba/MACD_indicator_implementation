#Amazon AMZN historyczne ceny, okres - dzienny

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# odczytywanie informacji z pliku
amazon_data = pd.read_csv('Historyczne ceny AMZN.csv')
amazon_data = amazon_data.iloc[::-1]
# usunięcie niepotrzebnych kolumn
amazon_data = amazon_data.drop(["Otwarcie", "Max.", "Min.", "Wol.", "Zmiana%"], axis=1)

# przerobienie tablicy odczytanych danych na typ danych float w kolumnie o nazwie "Ostatnio"
amazon_data["Ostatnio"] = amazon_data["Ostatnio"].astype(float)

# tablica próbek - kolumna "Ostatnio"
trials = np.array(amazon_data["Ostatnio"])

# rysowanie wykresu wizualizującego dane wejściowe
plt.plot(amazon_data['Ostatnio'])
# dodanie podpisów
plt.xlabel("Dzien")
plt.ylabel("Cena")
# dodanie nazwy wykresu
plt.title("Cena akcji Amazon w okresie 26.03.2019 - 15.03.2023")
plt.show()

# count alpha for 12 and 26 periods: a = 2/(N + 1) - liczenie alpha dla 12 i 26 okresów zgodnie ze wzorem
alpha12 = 2/(12 + 1)
alpha26 = 2/(26 + 1)

# array for ema12 - tablica dla EMA12
ema12 = []

# count EMA12 - liczenie EMA12 zgodnie ze wzorem i zapisywanie do tablicy
trials = trials[::-1]
for i in range(len(trials) - 12):
    ema12_top = trials[i]
    for j in range(1, 12 + 1): # exclusive
        ema12_top += ((1 - alpha12)**j)*trials[i + j]
    ema12_bottom = 1
    for j in range(1, 12 + 1):
        ema12_bottom += (1 - alpha12)**j
    ema12.append(ema12_top/ema12_bottom)

# count EMA26 - liczenie EMA26 zgodnie ze wzorem i zapisywanie do tablicy
ema26 = []
for i in range(len(trials) - 26):
    ema26_top = trials[i]
    for j in range(1, 26 + 1): # exclusive
        ema26_top += ((1 - alpha26)**j)*trials[i + j]
    ema26_bottom = 1
    for j in range(1, 26 + 1):
        ema26_bottom += (1 - alpha26)**j
    ema26.append(ema26_top/ema26_bottom)

# wyrównywanie rozmiarów tablic
while len(ema12) != len(trials):
    ema12.append(None)

while len(ema26) != len(trials):
    ema26.append(None)

# count MACD - liczenie MACD (len - 26, żeby nie było wychodzenia poza tablice -
# dlatego wyników jest mniej niż danych wejściowych)
macd = []
for i in range(len(trials) - 26):
    macd.append(ema12[i] - ema26[i])

# wyrównywanie rozmiarów tablic
while len(macd) != len(trials):
    macd.append(None)


# count signal - liczenie wskaźnika SIGNAL
alpha9 = 2/(9 + 1)
signal = []
for i in range(len(macd) - 26 - 9):
    signal_top = macd[i]
    for j in range(1, 9 + 1): # exclusive
        signal_top += ((1 - alpha9)**j)*macd[i + j]
    signal_bottom = 1
    for j in range(1, 9 + 1):
        signal_bottom += (1 - alpha9)**j
    signal.append(signal_top/signal_bottom)

# przepisywnie wartości MACD do nowej tablicy dla uproszczenia
macd_new = []
for i in range(966):
    macd_new.append(macd[i])
# rysowanie MACD na wykresie
plt.plot(np.array(macd_new), label="MACD")
# rysowanie SIGNAL na wykresie
plt.plot(np.array(signal), label="SIGNAL")
# rysowanie kropek w miejscach przecięcia MACD i SIGNAL za pomocą sprawdzania, czy się zmienia znak,
# czyli czy wykres w danym momencie zaczyna rosnąć czy maleć
idx = np.argwhere(np.diff(np.sign(np.array(macd_new) - np.array(signal)))).flatten()
plt.plot(np.arange(0, 1000)[idx], np.array(macd_new)[idx], 'ro')
# dodanie podpisu osi x
plt.xlabel("Dzien")
# dodanie nazwy wykresu
plt.title("MACD i SIGNAL")
# dodanie legendy
plt.legend(loc="upper left")
plt.show()
#print(idx)

macd_100 = []
signal_100 = []
amazon_data_50 = []
for i in range(50):
    macd_100.append(macd[i])
    signal_100.append(signal[i])
    amazon_data_50.append(amazon_data['Ostatnio'][i])

plt.plot(amazon_data_50)
# dodanie podpisów
plt.xlabel("Dzien")
plt.ylabel("Cena")
# dodanie nazwy wykresu
plt.title("Cena akcji Amazon w pierwsze 50 dni")
plt.show()
# 50 dni!!
# rysowanie MACD na wykresie
plt.plot(np.array(macd_100), label="MACD")
# rysowanie SIGNAL na wykresie
plt.plot(np.array(signal_100), label="SIGNAL")
# rysowanie kropek w miejscach przecięcia MACD i SIGNAL za pomocą sprawdzania, czy się zmienia znak,
# czyli czy wykres w danym momencie zaczyna rosnąć czy maleć
idx100 = np.argwhere(np.diff(np.sign(np.array(macd_100) - np.array(signal_100)))).flatten()
plt.plot(np.arange(0, 50)[idx100], np.array(macd_100)[idx100], 'ro')
# dodanie podpisu osi x
plt.xlabel("Dzien")
# dodanie nazwy wykresu
plt.title("MACD i SIGNAL w okresie 50 dni")
# dodanie legendy
plt.legend(loc="upper left")
plt.show()

# próba symulacji kupowania i sprzedawania akcji
budget = 1000
stocks = 0
for i in range(1, 966):
    if signal[i-1] - macd[i-1] >= 0 and ((signal[i] - macd[i])*(signal[i-1] - macd[i-1]) < 0) and budget - amazon_data['Ostatnio'][i] > 0:
        if macd[i] < 0:
            while budget > 100:
                budget -= amazon_data['Ostatnio'][i]
                stocks += 1
        print("kup")
        #print(i)
        print(budget)

    elif signal[i - 1] - macd[i - 1] <= 0 < stocks and ((signal[i] - macd[i]) * (signal[i - 1] - macd[i - 1]) < 0):
        if macd[i] > 0:
            budget += stocks*amazon_data['Ostatnio'][i]
            stocks = 0
        print("sprzedaj")
        #print(i)
        print(budget)

print(budget)