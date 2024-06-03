import matplotlib.pyplot as plt


def view(formula: str):
    try:
        plt.text(0.5, 0.5, '$' + formula.replace('\\%', '%').replace('%', '\\%') + '$', fontsize=30, ha='center')
        plt.axis('off') 
        plt.show()
    except:
        print('Error')

if __name__ == '__main__':
    view('E_k=\\dfrac{m\\cdot v^2}{2}')
