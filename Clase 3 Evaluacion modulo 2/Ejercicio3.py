import numpy as np 
def gen_rnd(n):
    List1 = []
    yi = int(np.random.randint(1,124578,1))
    for i in range (n):
        yi = (124578*yi+1)%(125)
        List1.append(yi)
    return List1
#generar lista
lista1 = gen_rnd(500)
print(lista1)
len(lista1)


lista = sorted(lista1)
print(lista)
len(lista)


#valores no repetidos
unicos = []
j = 0
while j < len(lista):
    if lista[j] not in unicos:
        unicos.append(lista[j])
    j += 1

len(unicos)
print(unicos)

#comprobacion
np.unique(lista1)
len(np.unique(lista1))



#multiplo de 7 o 13
mul7o13 = []

for i in range (0,len(unicos)):
    if unicos[i]%7 == 0 or unicos[i]%13 == 0:
        mul7o13.append(unicos[i])
print(mul7o13)


flags = {} 
def f(lista1):
    flags = {} 
    for i in range (1,lista1):
        if lista1[i]%i == 0:
                flags[i] = True
        else:
                flags[i] = False
    return flags

print(flags)


# pasar a diccionario primos TRUE y FALSE



def f(unicos)
    i = 0
    flags = {}
    m = True
    while i < len(unicos):
       
        if unicos[i] < 1:
            flags[unicos[i]] = False
        elif unicos[i]%2 == 0:
            flags[unicos[i]] = False
        else:
            j=1
            while j < unicos[i] and m == True:
                if unicos[i]%j == 0:
                    flags[unicos[i]] = False
                    m = False
            j += 1
        flags[unicos[i]] = True
    i += 1
len(flags)
flags

