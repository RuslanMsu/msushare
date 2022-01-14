from math import atan,exp,log
x0 = 2.436
def a1(x):
    if(x >=0 and x <= 2.436):
        return 1
    else:
        return -1

def R():
    return 1.6+1.2*(exp(-2*x0) -1) - (0.4/3.14)*(3.14 - atan(x0))
    
def a2(x_1,x_2):
    x_2_max = 0.05*(3*(17600*log(0.72*2.72**(8.7*(x_1 - 0.3)*(x_1+0.2)))+9)**(1/3)+19)
    x_2_min =  0.05*( 19 - 3*(17600*log(0.72*2.72**(8.7*(x_1 - 0.3)*(x_1+0.2)))+9)**(1/3))
    if(x_1 < -0.26 and x_2<x_2_max and x_2 > x_2_min ):
        return 1
    if(x_1 > 0.36 and x_2 < x_2_max and x_2 > x_2_min):
        return 1
    return -1
    
if __name__ == '__main__':
    print(a1(0.0), R(), a2(0.0, 0.0))
    
    with open('seminar02_task1.txt', 'w') as f:
        for i in range(-50, 50):
            x = i/10.0
            y = a1(x)
            f.write('%d ' % y)
            
    with open('seminar02_task2.txt', 'w') as f:
        f.write('%.3f' % R())   
    
    with open('seminar02_task3.txt', 'w') as f:
        for i in range(-50, 50):
            x1 = i/10.0
            for j in range(-50, 50):
                x2 = j/10.0
                y = a2(x1, x2)
                f.write('%d ' % y) 