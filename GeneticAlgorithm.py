import numpy as np
import matplotlib.pyplot as plt
import random
import time
import cv2

class AdThreshold():
    def __init__(self, image, HSV=0):
        #HSV为3时，彩色图像灰度化处理。HSV为0、1、2分别对应着使用H、S、V通道处理, HSV为4时，直方图是所有通道的叠加求得的,并使用灰度图像求mask.
        self.image = image
        self.height=image.shape[0]
        self.width=image.shape[1]
        self.HSV = HSV
        if(len(image.shape)==2):
            self.isGray=1
            self.channels_num=1
            self.hist = self.__myhist(image)[0]
        else:
            self.isGray=0
            self.image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            self.channels_num=image.shape[2]
            if(HSV==4): #所有通道一起叠加的直方图来求阈值
                self.hist = np.sum(self.__myhist(self.image), axis=0)
            elif(HSV==3): #彩色图像先灰度化在求mask
                self.hist = self.__myhist(cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY))[0]
            else:
                self.hist = self.__myhist(self.image_HSV[:,:,HSV])[0]
        assert self.hist.shape==(256,)

    def Basic(self, test=0, eps=1e-5):
        # 注意Basic是使用图像的中间灰度值初始化阈值的，对于分别不均匀的图片有可能收敛不到！！
        if(self.isGray):
            image = self.image
        else:
            if(self.HSV==3 or self.HSV==4):
                image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                image = self.image_HSV[:,:,self.HSV]
        if(test):
            print("最小灰度为: ", np.min(image)/2)
            print("中间灰度为: ", np.max(image)/2+np.min(image)/2)
            thres0_temp = np.max(image)-5
        else:
            thres0_temp = np.round(np.max(image)/2+np.min(image)/2) # 不能先加再除以2，因为是uint8类型，直接帮你模2加了！！
        image_c1 = image*(image < thres0_temp)
        image_c2 = image*(image >= thres0_temp)
        mean_bg = np.sum(image_c1)/np.sum(image < thres0_temp)
        mean_fg = np.sum(image_c2)/np.sum(image >= thres0_temp)
        thres0 = np.round(mean_bg/2 + mean_fg/2)
        while(np.abs(thres0-thres0_temp)>eps):
            thres0_temp = thres0
            image_c1 = image*(image < thres0)
            image_c2 = image*(image >= thres0)
            mean_bg = np.sum(image_c1)/np.sum(image < thres0)
            mean_fg = np.sum(image_c2)/np.sum(image >= thres0)
            thres0 = np.round(mean_bg/2 + mean_fg/2)
        self.mean_bg = mean_bg
        self.mean_fg = mean_fg
        self.thres = thres0
        self.mask = image>=thres0
        if(self.isGray):
            self.image_class = [self.mask*self.image, (~self.mask)*self.image]
        else:
            self.image_class = [self.mask[:,:,np.newaxis]*self.image, (~self.mask[:,:,np.newaxis])*self.image]
        #self.visualization(image, thres0, 'Basic')
        print('Basic 的阈值为%d'%(thres0))
    
    def Ostu(self):
        if(self.isGray):
            image = self.image
        else:
            if(self.HSV==3 or self.HSV==4):
                image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                image = self.image_HSV[:,:,self.HSV]
        delta_max = 0
        thres_best = np.inf
        for thres in range(256):
            P1 = np.sum(self.hist[:thres])/(self.height*self.width)
            P2 = np.sum(self.hist[thres:])/(self.height*self.width)
            m1 = np.sum([i*self.hist[i] for i in range(thres)])/(np.sum(self.hist[:thres])+1e-6)
            m2 = np.sum([i*self.hist[i] for i in range(thres,256)])/(np.sum(self.hist[thres:])+1e-6)
            mG = P1*m1+P2*m2
            delta = P1*(m1-mG)**2+P2*(m2-mG)**2
            if delta>delta_max:
                thres_best = thres
                delta_max  = delta
        self.thres = thres_best
        self.mask = image>=thres_best
        if(self.isGray):
            self.image_class = [self.mask*self.image, (~self.mask)*self.image]
        else:
            self.image_class = [self.mask[:,:,np.newaxis]*self.image, (~self.mask[:,:,np.newaxis])*self.image]
        #self.visualization(image, thres_best, 'Ostu')
        print('Ostu 的阈值为%d'%(thres_best))

    def MOstu(self):
        #多阈值的Ostu
        if(self.isGray):
                image = self.image
                self.image_class = np.zeros((self.k,self.height,self.width), dtype=np.uint8)
        else:
            self.image_class = np.zeros((self.k,self.height,self.width,3), dtype=np.uint8)
            if(self.HSV==3 or self.HSV==4):
                image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                image = self.image_HSV[:,:,self.HSV]
        assert(len(self.thres)==self.k-1)
        breakpoint = np.concatenate(([0], self.thres, [255]))
        assert(len(breakpoint)==self.k+1)
        mask = np.zeros((self.k,self.height,self.width))
        for i in range(self.k):
            mask[i] = (image>=breakpoint[i]) * (image<breakpoint[i+1])
            if(self.isGray):
                self.image_class[i] = mask[i][:,:]*self.image
            else:
                self.image_class[i] = mask[i][:,:,np.newaxis]*self.image
        self.mask = mask
        if(self.isGray):
            for j in range(self.k):
                self.image_class[j] = self.mask[j]*self.image
        else:
            for j in range(self.k):
                self.image_class[j] = self.mask[j][:,:,np.newaxis]*self.image


    def visualization(self, title, mostu=0):
        # title是窗口标题名，mostu为1表示使用的是多阈值Ostu
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(self.image, 'gray')
        plt.title('image')
        if(mostu==0):
            plt.subplot(2,2,2)
            plt.imshow(255*(self.mask), 'Blues')
            plt.title(title+' mask')
            plt.subplot(2,2,3)
            plt.imshow(self.image_class[0], 'gray')
            plt.title('class1')
            plt.subplot(2,2,4)
            plt.imshow(self.image_class[1], 'gray')
            plt.title('class2')
        else:
            if(self.k<=3):
                plt.figure()
                for i in range(1, self.k+1):
                    plt.subplot(1,self.k,i)
                    plt.imshow(255*(self.mask[i-1]), 'Blues')
                    plt.title(title+' mask')
                plt.figure()
                for i in range(1, self.k+1):
                    plt.subplot(1,self.k,i)
                    plt.imshow(self.image_class[i-1], 'gray')
                    plt.title(title+' class'+str(i))
            else:
                plt.figure()
                for i in range(1, self.k+1):
                    plt.subplot(2, self.k//2+self.k%2, i)
                    plt.imshow(255*(self.mask[i-1]), 'Blues')
                    plt.title(title+' mask')
                plt.figure()                
                for i in range(1, self.k+1):
                    plt.subplot(2, self.k//2+self.k%2, i)
                    plt.imshow(self.image_class[i-1], 'gray')
                    plt.title(title+' class'+str(i))

    
    def plotHist(self, title, Basic=0):
        #绘制直方图，Basic为1表示使用的是基本全局阈值处理，此时会将最后迭代的两类各自的平均灰度以黑色虚线画出
        plt.figure()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title(title)
        ax.bar(range(0,256), self.hist)
        if(type(self.thres)==list):
            for th in self.thres:
                ax.vlines(th, 0, np.max(self.hist), linestyles='dashed', colors='red')
        else: ax.vlines(self.thres, 0, np.max(self.hist), linestyles='dashed', colors='red')
        if(Basic):
            ax.vlines(self.mean_bg, 0, np.max(self.hist), linestyles='dashed', colors='black')
            ax.vlines(self.mean_fg, 0, np.max(self.hist), linestyles='dashed', colors='black')

    def __myhist(self, image):
        # 计算图像直方图，image为对象图像或对象均衡化后图像
        if(len(image.shape)==2):
            hist = np.zeros([1, 256], int)
            max_num = np.max(image)
            min_num = np.min(image)
            for i in range(min_num, max_num+1):
                hist[0, i] = np.sum(image==i)
            assert np.sum(hist)==self.height*self.width
            return hist
        else:
            hist = np.zeros([self.channels_num, 256], int)
            for ch in range(self.channels_num): 
                hist[ch,:] = self.__myhist(image[:,:,ch])
        return hist


class GA(AdThreshold):
    #初始化种群
    def __init__(self, image, pop_size=50, k=3, P_cross=0.5, cross_num=2, P_mutation=0.1, adaption=1, Generation=100, elite_num=3, HSV=3):
    #输入：种群大小和类别数量
    #输出：pop_size*(k-1)*8的种群矩阵，每一行代表一个个体
    #个体：一个长度为(k-1)*8的01序列，每8位是一个阈值的二进制表示
    #种群：由pop_size个个体组成
        super(GA, self).__init__(image, HSV)
        self.pop_size = pop_size
        self.k = k
        self.population = (np.random.randint(0, 2, (pop_size, (k-1)*8))).tolist()
        #self.hist = self.hist
        self.P_cross = P_cross       # 交叉概率
        self.cross_num = cross_num   # 交叉点数
        self.P_mutation = P_mutation # 变异概率
        self.adaption = adaption     #自适应交叉变异概率
        self.Generation = Generation #迭代代数
        self.elite_num = elite_num   #精英解保持数

    def __list2thres(self, individual):
        thres_ind = []
        for bp in range(self.k-1):
            thres = 0
            thres_bin = individual[bp*8:(bp+1)*8]
            for i in range(8):
                thres += thres_bin[i]*(2**(7-i))
            thres_ind.append(thres)
        assert len(thres_ind)==self.k-1
        return thres_ind

    #适应度计算
    def Fitnesscal(self, mixpopulation):
    # 计算mixpopulation的适应度，并更新self.fitness
        pop_size = len(mixpopulation)
        delta = np.zeros(pop_size)
        for i in range(pop_size):
            # 对每一个个体
            breakpoint = np.zeros(self.k+1, dtype=np.uint8)
            Pj = np.zeros(self.k) # 类概率
            mj = np.zeros(self.k) # 类平均灰度
            # 先求出所有类之间的断点
            thres = self.__list2thres(mixpopulation[i])
            breakpoint[1:-1] = thres
            breakpoint[0] = 0; breakpoint[-1] = 255;
            breakpoint = np.sort(breakpoint)
            for j in range(self.k):
                # 对每一个类
                hist_j = self.hist[breakpoint[j]:breakpoint[j+1]]
                Pj[j] = np.sum(hist_j)/(self.height*self.width)
                mj[j] = np.sum(range(breakpoint[j],breakpoint[j+1])*hist_j)/(np.sum(hist_j)+1e-6)

            mG = np.sum(Pj*mj)
            delta[i] = np.sum(Pj*((mj-mG)**2)) #类间方差
        self.fitness = delta.tolist()
        assert len(self.fitness)==pop_size
        #return delta

    #选择
    def Select(self, mixpopulation, method = "roulette", tour_order = 10):
    # 目的从mixpopulation中挑选优秀的个体生存，更新self.population
    # method是选择方法，有轮盘赌和锦标赛（tour_order是锦标赛阶数）
        pop_size = self.pop_size-self.elite_num
        fitness_sum = np.sum(self.fitness)
        P_select = self.fitness/fitness_sum #选中概率
        P_acc = np.cumsum(P_select) #累计概率
        population_new = []
        if(method == "roulette"):# 轮盘赌
            for j in range(pop_size):
                r = np.random.uniform(0,1)
                for i in range(len(P_acc)):
                    if(i == 0):
                        if r >= 0 and r <= P_acc[i]:
                            population_new.append(mixpopulation[i])
                    elif(r > P_acc[-1]):
                        print('由于舍入误差，累积概率和略小于1，此处出现了罕见的随机数大于该和的现象')
                        population_new.append(mixpopulation[-1])
                    else:
                        if r > P_acc[i-1] and r <= P_acc[i]:
                            population_new.append(mixpopulation[i])
                            break
                    
        elif(method == "tournament"):# 锦标赛
            for j in range(pop_size):
                r = random.sample(range(pop_size), tour_order)
                for i in range(tour_order): #将入选个体取出
                    fitness_can.append(self.fitness[r[i]])
                    population_can.append(mixpopulation[r[i]])
                population_new.append(population_can[lexsort(fitness_can)[-1]])
                #将候选个体中适应度最大的放入子代    
        assert(len(population_new)==pop_size)
        return population_new

    #交叉
    def Crossover(self):
    #功能：产生子代，更新self.offspring
        mid = int(self.pop_size/2)    #一半做父本，一半做母本
        np.random.shuffle(self.population)   #打乱父本，随机交叉
        father = self.population[:mid]       #父本
        mother = self.population[mid:]       #母本
        goods_num = (self.k-1)*8             #个体(染色体长度)
        offspring = []                       #子代种群
        f_best = np.max(self.fitness)
        f_mean = np.mean(self.fitness)
        for i in range(mid):
            #自适应调整交叉概率
            if(self.adaption):
                f_better = max(self.fitness[i], self.fitness[mid+i])
                if(f_better >= f_mean):
                    P_cross_now = self.P_cross*(f_best - f_better)/(f_best - f_mean)
                else:
                    P_cross_now = self.P_cross
            else:
                P_cross_now = self.P_cross
            r = np.random.uniform(0,1)
            if r <= P_cross_now:
                #多点交叉
                cross_index = np.sort(random.sample(range(1, goods_num), self.cross_num))
                #在一个个体(染色体)的中间生成cross_num个交叉点
                offspring1 = []; offspring2 = []
                for j in range(self.cross_num): #对于每个子代，奇数偶数交叉点依次取父本母本的基因
                    if(j == 0):
                        offspring1 = offspring1 + father[i][:cross_index[j]]
                        offspring2 = offspring2 + mother[i][:cross_index[j]]
                        if(j == self.cross_num-1): #补上最后一段的交叉基因
                            offspring1 = offspring1 + mother[i][cross_index[j]:]
                            offspring2 = offspring2 + father[i][cross_index[j]:]
                    elif(j%2 == 1):
                        offspring1 = offspring1 + mother[i][cross_index[j-1]:cross_index[j]]
                        offspring2 = offspring2 + father[i][cross_index[j-1]:cross_index[j]]
                        if(j == self.cross_num-1): #补上最后一段的交叉基因
                            offspring1 = offspring1 + father[i][cross_index[j]:]
                            offspring2 = offspring2 + mother[i][cross_index[j]:]
                    else:
                        offspring1 = offspring1 + father[i][cross_index[j-1]:cross_index[j]]
                        offspring2 = offspring2 + mother[i][cross_index[j-1]:cross_index[j]]
                        if(j == self.cross_num-1): #补上最后一段的交叉基因
                            offspring1 = offspring1 + mother[i][cross_index[j]:]
                            offspring2 = offspring2 + father[i][cross_index[j]:]
                assert(len(offspring1)==goods_num)
                assert(len(offspring2)==goods_num)            

            else: #不交叉
                offspring1 = father[i]
                offspring2 = mother[i]
            offspring.append(offspring1)
            offspring.append(offspring2)
        self.offspring = offspring

    #变异
    def Mutation(self):
    # 更新变异后的子代
        for i in range(len(self.offspring)):
            if(self.adaption):
                f_best = np.max(self.fitness)
                f_mean = np.mean(self.fitness)
                if(self.fitness[i] >= f_mean):
                    P_mut_now = self.P_mutation*(f_best - self.fitness[i])/(f_best - f_mean)
                else:
                    P_mut_now = self.P_mutation
            else:
                P_mut_now = self.P_mutation
            r=np.random.uniform(0,1)
            if r <= P_mut_now:
                mut_point = np.random.randint(0, 8*(self.k-1))
                self.offspring[i][mut_point] = 1-self.offspring[i][mut_point]      
    
    def forward(self):
        self.Fitnesscal(self.population) # self.fitness
        elite_gen_fitness = [np.max(self.fitness)]
        elite_gen = [self.population[self.fitness.index(np.max(self.fitness))]]
        
        #繁衍开始
        start = time.time()
        for i in range(self.Generation):
            if(i%(self.Generation/10) == 0):
                print("迭代进度：%.2f%%"%(i/self.Generation*100))

            #交叉
            self.Crossover() # 更新self.offspring
            #变异
            self.Mutation()  # 更新self.offspring
            mixpopulation = np.concatenate([self.population, self.offspring])
            #适应度函数计算
            self.Fitnesscal(mixpopulation)
            mixpopulation = mixpopulation.tolist()
            #精英解保持
            fitness_where = np.lexsort([self.fitness])[::-1]
            for index in range(self.elite_num):
                self.population[index] = mixpopulation[fitness_where[index]]
            assert(self.fitness[fitness_where[0]] == np.max(self.fitness))
            #剔除精英
            fitness_where = np.sort(fitness_where[:self.elite_num])
            for j in range(self.elite_num):
                self.fitness.pop(fitness_where[j]-j)
                mixpopulation.pop(fitness_where[j]-j) 
            #轮盘赌选择
            self.population[self.elite_num:] = self.Select(mixpopulation, method = "roulette")
            assert(len(self.population)==self.pop_size)
            #记录每一代的最优解
            self.Fitnesscal(self.population)
            elite_gen_fitness.append(np.max(self.fitness))
            maxfit_index = self.fitness.index(np.max(self.fitness))
            elite_gen.append(self.population[maxfit_index])  
        end = time.time()
        #寻找最优
        assert(len(elite_gen_fitness)==self.Generation+1)
        best_fitness = np.max(elite_gen_fitness)                      #最优解的适应度
        best_gen_temp = np.where(elite_gen_fitness == best_fitness)   #最优解出现在第几代
        best_gen_th = best_gen_temp[0][0]
        best_ind = elite_gen[best_gen_th][:]                          #最优解的染色体
        allthres = np.ones(self.k-1)
        # 求出阈值
        thres = self.__list2thres(best_ind)
        thres = np.sort(thres)
        self.thres = thres
        #输出结果
        print("分成%d类"%(self.k))
        print("精英解保持：%d"%(self.elite_num))
        print("各个阈值为：", thres)
        print("最大方差为：%d"%(best_fitness))
        print("测试用时：%f"%(end-start))
        print("最优解出现的代数为：%d"%(best_gen_th))

        #画出收敛曲线
        plt.figure(0)
        plt.plot(elite_gen_fitness)
        plt.xlabel('iterations')
        plt.ylabel('the elite of each Generationeration')
