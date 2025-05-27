import numpy as np

def crear_matriz_K(n, k_const=1.0):
    """
    Construye la matriz K para un sistema n x n con constantes de resorte iguales.
    Se conecta cada masa con sus vecinos (arriba, abajo, izquierda, derecha) si existen.
    - n: tamaño de la malla (n x n)
    - k_const: valor constante de cada resorte (por simplicidad)
    
    Retorna:
    - K: matriz 2n^2 x 2n^2 (numpy array)
    """
    N = n * n
    dim = 2 * N
    K = np.zeros((dim, dim))
    
    # Función auxiliar para obtener el índice del vector R (x o y) para masa en (i,j)
    def idx(i, j, coord):  
        # coord=0 para x, coord=1 para y
        masa = i * n + j
        return 2 * masa + coord

    for i in range(n):
        for j in range(n):
            for coord in [0,1]:  # x=0, y=1
            
                diagonal_pos = idx(i,j,coord)
                suma_k_vecinos = 0
                
                # Vecinos y sus constantes
                vecinos = []
                
                # Arriba (i+1, j)
                if i+1 < n:
                    vecinos.append(idx(i+1,j,coord))
                    suma_k_vecinos += k_const
                # Abajo (i-1, j)
                if i-1 >= 0:
                    vecinos.append(idx(i-1,j,coord))
                    suma_k_vecinos += k_const
                # Derecha (i, j+1)
                if j+1 < n:
                    vecinos.append(idx(i,j+1,coord))
                    suma_k_vecinos += k_const
                # Izquierda (i, j-1)
                if j-1 >= 0:
                    vecinos.append(idx(i,j-1,coord))
                    suma_k_vecinos += k_const
                
                # Asignar diagonal
                K[diagonal_pos, diagonal_pos] = suma_k_vecinos
                
                # Asignar elementos fuera de diagonal (vecinos)
                for v in vecinos:
                    K[diagonal_pos, v] = -k_const

    return K

def main():
    n = 3  # Tamaño de la malla 3x3 para prueba
    k = 1.0  # Constante de resorte igual para todos
    
    K = crear_matriz_K(n, k)
    
    print("Matriz K:")
    print(K)
    
    # Calcular valores y vectores propios
    eigvals, eigvecs = np.linalg.eig(K)
    
    # Ordenar por valor absoluto de los autovalores para mejor interpretación
    idx_sorted = np.argsort(np.abs(eigvals))
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]
    
    print("\nPrimeros valores propios (λ):")
    print(eigvals[:6])
    
    # Calcular frecuencias omega = sqrt(-lambda) (solo si lambda negativo)
    omega = np.array([np.sqrt(-l) if l < 0 else 0.0 for l in eigvals])
    print("\nPrimeras frecuencias omega:")
    print(omega[:6])

if __name__ == "__main__":
    main()
