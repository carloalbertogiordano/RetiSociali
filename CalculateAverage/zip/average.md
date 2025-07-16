Tutti gli algoritmi (e le loro combinazioni) sono stati eseguiti 5 volte per poterne calcolare una media dei risultati
# Cost-Seeds Greedy
L'algoritmo CSG è stato eseguito 5 volte per ogni combinazione di $\{c_1,c_2,c_3\}\times\{f_1,f_2,f_3\}$

## Cost function $c_1(u)$
La funzione $c_1(u)$ è la seguente:
$$
c_1(u)&=\text{valore casuale in un range fissato}
$$

### Funzione obiettifo $f_1(S)$
La funzione obiettivo $f_1(S)$ è la seguente:
$$
f_1(S) &= \sum_{v\in V} \min\!\Bigl\{|N(v)\cap S|,\;\bigl\lceil \tfrac{d(v)}{2} \bigr\rceil\Bigr\}
$$

### Results:
📁 CSG_random_f1.json ➜  ΔMedia Cascade - SS: 72.20
---

### Funzione obiettifo $f_2(S)$
La funzione obiettivo $f_2(S)$ è la seguente:
$$
f_2(S) &= \sum_{v\in V}\sum_{i=1}^{|N(v)\cap S|} \max\!\Bigl\{\bigl\lceil\tfrac{d(v)}{2}\bigr\rceil - i + 1,\;0\Bigr\}
$$

### Results:
📁 CSG_random_f2.json ➜  ΔMedia Cascade - SS: 22.40
---

### Funzione obiettifo $f_3(S)$
La funzione obiettivo $f_3(S)$ è la seguente:
$$
f_3(S) &= \sum_{v\in V}\sum_{i=1}^{|N(v)\cap S|} \max\!\Bigl\{\frac{\bigl\lceil\tfrac{d(v)}{2}\bigr\rceil - i + 1}{d(v)-i+1},\;0\Bigr\}
$$

### Results:
📁 CSG_random_f3.json ➜  ΔMedia Cascade - SS: 90.20
---

## Cost function $c_2(u)$
La funzione $c_2(u)$ è la seguente:
$$
c_2(u)&=\bigl\lceil \tfrac{d(u)}{2} \bigr\rceil
$$
### Funzione obiettifo $f_1(S)$
La funzione obiettivo $f_1(S)$ è la seguente:
$$
f_1(S) &= \sum_{v\in V} \min\!\Bigl\{|N(v)\cap S|,\;\bigl\lceil \tfrac{d(v)}{2} \bigr\rceil\Bigr\}
$$

#### Results:
📁 CSG_degree_f1.json ➜  ΔMedia Cascade - SS: 199.00
---

### Funzione obiettifo $f_2(S)$
La funzione obiettivo $f_2(S)$ è la seguente:
$$
f_2(S) &= \sum_{v\in V}\sum_{i=1}^{|N(v)\cap S|} \max\!\Bigl\{\bigl\lceil\tfrac{d(v)}{2}\bigr\rceil - i + 1,\;0\Bigr\}
$$
 
### Results:
📁 CSG_degree_f2.json ➜  ΔMedia Cascade - SS: 3.00
---

### Funzione obiettifo $f_3(S)$
La funzione obiettivo $f_3(S)$ è la seguente:
$$
f_3(S) &= \sum_{v\in V}\sum_{i=1}^{|N(v)\cap S|} \max\!\Bigl\{\frac{\bigl\lceil\tfrac{d(v)}{2}\bigr\rceil - i + 1}{d(v)-i+1},\;0\Bigr\}
$$


### Results:
📁 CSG_degree_f3.json ➜  ΔMedia Cascade - SS: 158.00
---

## Cost function $c_3(u)$
La funzione $c_3(u)$ è la seguente:
$$
c_3(u)&=\Bigl\lceil \tfrac{\displaystyle\sum_{v\in N(u)} \frac{d(v)}{|N(v)\cap N(u)|+1}}{2}\Bigr\rceil
$$
### Funzione obiettifo $f_1(S)$
La funzione obiettivo $f_1(S)$ è la seguente:
$$
f_1(S) &= \sum_{v\in V} \min\!\Bigl\{|N(v)\cap S|,\;\bigl\lceil \tfrac{d(v)}{2} \bigr\rceil\Bigr\}
$$

### Results:
📁 CSG_custom_f1.json ➜  ΔMedia Cascade - SS: 76.00
---

### Funzione obiettifo $f_2(S)$
La funzione obiettivo $f_2(S)$ è la seguente:
$$
f_2(S) &= \sum_{v\in V}\sum_{i=1}^{|N(v)\cap S|} \max\!\Bigl\{\bigl\lceil\tfrac{d(v)}{2}\bigr\rceil - i + 1,\;0\Bigr\}
$$
 
### Results:
📁 CSG_custom_f2.json ➜  ΔMedia Cascade - SS: 0.00
---

### Funzione obiettifo $f_3(S)$
La funzione obiettivo $f_3(S)$ è la seguente:
$$
f_3(S) &= \sum_{v\in V}\sum_{i=1}^{|N(v)\cap S|} \max\!\Bigl\{\frac{\bigl\lceil\tfrac{d(v)}{2}\bigr\rceil - i + 1}{d(v)-i+1},\;0\Bigr\}
$$


### Results:
📁 CSG_custom_f3.json ➜  ΔMedia Cascade - SS: 86.00
---

# WTSS
L'algoritmo WTSS è stato eseguito 5 volte per ogni funzione di costo $\{c_1,c_2,c_3\}$

## Cost function $c_1(u)$
La funzione $c_1(u)$ è la seguente:
$$
c_1(u)&=\text{valore casuale in un range fissato}
$$

### Results
📁 WTSS_random_none.json ➜  ΔMedia Cascade - SS: 3124.00
---

## Cost function $c_2(u)$
La funzione $c_2(u)$ è la seguente:
$$
c_2(u)&=\bigl\lceil \tfrac{d(u)}{2} \bigr\rceil
$$

### Results
📁 WTSS_degree_none.json ➜  ΔMedia Cascade - SS: 97.00
---

## Cost function $c_3(u)$
La funzione $c_3(u)$ è la seguente:
$$
c_3(u)&=\Bigl\lceil \tfrac{\displaystyle\sum_{v\in N(u)} \frac{d(v)}{|N(v)\cap N(u)|+1}}{2}\Bigr\rceil
$$

### Results
📁 WTSS_custom_none.json ➜  ΔMedia Cascade - SS: 2756.00
---

# GA-CMC
L'algoritmo GA-CMC è stato eseguito 5 volte per ogni funzione di costo $\{c_1,c_2,c_3\}$

## Cost function $c_1(u)$
La funzione $c_1(u)$ è la seguente:
$$
c_1(u)&=\text{valore casuale in un range fissato}
$$

### Results
📁 GENETIC_random_none.json ➜  ΔMedia Cascade - SS: 50.60
---

## Cost function $c_2(u)$
La funzione $c_2(u)$ è la seguente:
$$
c_2(u)&=\bigl\lceil \tfrac{d(u)}{2} \bigr\rceil
$$

### Results
📁 GENETIC_degree_none.json ➜  ΔMedia Cascade - SS: 174.20
---

## Cost function $c_3(u)$
La funzione $c_3(u)$ è la seguente:
$$
c_3(u)&=\Bigl\lceil \tfrac{\displaystyle\sum_{v\in N(u)} \frac{d(v)}{|N(v)\cap N(u)|+1}}{2}\Bigr\rceil
$$

### Results
📁 GENETIC_custom_none.json ➜  ΔMedia Cascade - SS: 84.20
---

## GA con budget più alto e 300 pop_size
### Results
📁 GENETIC_custom_none.json ➜  ΔMedia Cascade - SS: 150.20
📁 GENETIC_degree_none.json ➜  ΔMedia Cascade - SS: 1990.40
📁 GENETIC_random_none.json ➜  ΔMedia Cascade - SS: 114.40

## GA con budget più alto e 400 pop_size
### Results
📁 GENETIC_custom_none.json ➜  ΔMedia Cascade - SS: 186.20
📁 GENETIC_degree_none.json ➜  ΔMedia Cascade - SS: 2138.20
📁 GENETIC_random_none.json ➜  ΔMedia Cascade - SS: 129.80

## GA con budget più basso e 500 pop_size 
📁 GENETIC_custom_none.json ➜  ΔMedia Cascade - SS: 3.20
📁 GENETIC_degree_none.json ➜  ΔMedia Cascade - SS: 6.00
📁 GENETIC_random_none.json ➜  ΔMedia Cascade - SS: 0.20

## GA con budget più alto e 500 pop_size
📁 GENETIC_custom_none.json ➜  ΔMedia Cascade - SS: 233.80
📁 GENETIC_degree_none.json ➜  ΔMedia Cascade - SS: 2009.80
📁 GENETIC_random_none.json ➜  ΔMedia Cascade - SS: 88.60