import sys 

C = float(sys.argv[1])
F = 9/5*C+32

print("""
      Temp [Celsius] : %f
      Temp [Farenheit] : %f
      """%(C,F))

# Recordar que al usar el modulo
# sys , esto obliga a tener el 
# codigo en un solo archivo de 
# extension py