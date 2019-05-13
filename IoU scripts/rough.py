import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Python')
y_pos = [1]
performance = 10
 
plt.bar(y_pos, performance, align='center', alpha=0.5, width=0.1)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')
 
plt.show()