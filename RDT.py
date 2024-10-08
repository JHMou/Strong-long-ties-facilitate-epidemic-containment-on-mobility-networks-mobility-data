import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import r2_score
def SEIR_Reaction_Diffusion(Grid_SEIR, P, c, elong, r,v,D):  # 先反应再扩散,c,elong,r分别表示S到E，E到I，I到R的概率
    #记录每日新增
    S_i = []
    E_i = []
    I_i = []
    R_i = []
    N_i = []
    deta_I=[]
    P1 = P.copy()
    for i in range(Grid_SEIR.shape[0]):
        if Grid_SEIR.loc[i][4] == 0:  
            S_i.append(0)
            E_i.append(0)
            I_i.append(0)
            R_i.append(0)
            N_i.append(0)
            deta_I.append(0)
        else:
            S_i1 = Grid_SEIR.loc[i][0]
            E_i1 = Grid_SEIR.loc[i][1]
            I_i1 = Grid_SEIR.loc[i][2]
            R_i1 = Grid_SEIR.loc[i][3]
            N_i1 = Grid_SEIR.loc[i][4]

            dS_dt1 =  (1 - D) * (S_i1 - c * S_i1 * I_i1 / N_i1-v*S_i1 ) # 自身粒子情况
            dE_dt1 =  (1 - D) * (c * S_i1 * I_i1 / N_i1 + (1 - elong) * E_i1)
            dI_dt1 = (1 - D) * (elong * E_i1 + (1 - r) * I_i1)
            dR_dt1 =  (1 - D) * (r * I_i1 + R_i1+v*S_i1)
            deta_I1=elong * E_i1#从E到I，不考虑从其他grid转移，转移并没有新增

            # 其他grid的转移粒子
            S_j1 = np.array(Grid_SEIR.S)
            E_j1 = np.array(Grid_SEIR.E)
            I_j1 = np.array(Grid_SEIR.I)
            R_j1 = np.array(Grid_SEIR.R)
            N_j1 = S_j1 + E_j1 + I_j1 + R_j1
            p_j_i1 = P1[:, i]
            MID1 = {'S': list(S_j1), 'E': list(E_j1), 'I': list(I_j1), 'R': list(R_j1), 'N': list(N_j1),
                    'P': list(p_j_i1)}
            MID2 = pd.DataFrame(MID1)
            MID = MID2[MID2.N != 0]
            S_j = np.array(MID.S)
            E_j = np.array(MID.E)
            I_j = np.array(MID.I)
            R_j = np.array(MID.R)
            N_j = S_j + E_j + I_j + R_j
            p_j_i = np.array(MID.P)

            S_list = p_j_i * (S_j - c * S_j * I_j / N_j-v*S_j)  # array操作
            dS_dt2 = D * sum(S_list)-S_i1
            E_list = p_j_i * ((1 - elong) * E_j + c * S_j * I_j / N_j)
            dE_dt2 = D * sum(E_list)-E_i1
            I_list = p_j_i * (elong * E_j + (1 - r) * I_j)
            dI_dt2 = D * sum(I_list)-I_i1
            R_list = p_j_i * (r * I_j + R_j+v*S_j)
            dR_dt2 = D * sum(R_list)-R_i1

            S_i.append(S_i1 + dS_dt1 + dS_dt2)
            E_i.append(E_i1 + dE_dt1 + dE_dt2)
            I_i.append(I_i1 + dI_dt1 + dI_dt2)
            R_i.append(R_i1 + dR_dt1 + dR_dt2)
            deta_I.append(deta_I1)

    Grid_SEIR['S'] = S_i
    Grid_SEIR['E'] = E_i
    Grid_SEIR['I'] = I_i
    Grid_SEIR['R'] = R_i
    Grid_SEIR['deta_I'] = deta_I
    Grid_SEIR['N'] = list(np.array(S_i) + np.array(E_i) + np.array(I_i) + np.array(R_i))  # 每个grid的人口在变化

    return Grid_SEIR  # 返回t+1时刻SEIR数据，Grid_SEIR表示累积数量



def simulation(Grid_SEIR,v_after,D_after):
    c = 3.4/5.6
    elong = 1 / 1.2
    r = 1 / 5.6
    icount = 1
    Grid_SEIR_T = Grid_SEIR.copy()
    I = []
    D_I=[]#新增病例
    grid_num=[]
    file=open('Trans_P_封城前.pickle','rb')
    P_before=pickle.load(file)
    file.close()
    file = open('Trans_P_0328-0531.pickle', 'rb')
    P_after = pickle.load(file)
    file.close()
    v_before=0
    D_before=1
    if True:
        while icount < 90:
            if icount<=26:
                P=P_before
                D=D_before
                v=v_before
            else:
                P = P_after
                D = D_after
                v = v_after
            Grid_SEIR_T = SEIR_Reaction_Diffusion(Grid_SEIR_T, P, c, elong, r, v,D)

            I.append(sum(Grid_SEIR_T.I))
            D_I.append(sum(Grid_SEIR_T.deta_I))
            Grid_SEIR_T1=Grid_SEIR_T[Grid_SEIR_T.I != 0]
            grid_num.append(Grid_SEIR_T1.shape[0])
            print(str(icount) + 'epoch:' + str(I[-1]))
            print('grid:'+str(grid_num[-1]))
            icount += 1


        file = open(f'./result/I_{v_after}_{D_after}.pickle', 'wb')
        pickle.dump(I, file)
        file.close()

        file = open(f'./result/DETA_I_{v_after}_{D_after}.pickle', 'wb')
        pickle.dump(D_I, file)
        file.close()

        file = open(f'./result/grid_num_{v_after}_{D_after}.pickle', 'wb')
        pickle.dump(grid_num, file)
        file.close()
        return D_I

if __name__=='__main__':

    case_data1 = pd.read_csv('case_data.csv')
    grid_rect = pd.read_csv('shanghai_grid.csv')

    case_data = case_data1.dropna(axis=0, subset=["location_bd09ll"])  # 删除位置信息列中空值对应行
    case_data['time'] = pd.to_datetime(case_data.date)
    case_data.sort_values(by=['time'], ascending=True, inplace=True)
    case_data['long'] = case_data['location_bd09ll'].apply(lambda x: float(x.split(',')[0]))
    case_data['lat'] = case_data['location_bd09ll'].apply(lambda x: float(x.split(',')[1]))
    start_time = '2022-03-01'
    end_time = '2022-03-05'
    case_data2 = case_data[(case_data.time >= pd.to_datetime(start_time)) & (case_data.time <= pd.to_datetime(end_time))]
    for igrid in range(grid_rect.shape[0]):
        num = case_data2[ (case_data2.long >= grid_rect.iloc[igrid, 1]) & (case_data2.lat >= grid_rect.iloc[igrid, 2]) & (case_data2.long < grid_rect.iloc[igrid, 5]) & (case_data2.lat < grid_rect.iloc[igrid, 6])].count()
        grid_rect.loc[igrid, 'patient'] = num[0]
    # grid_rect每个grid里面的特定时间区间内的病例数量，作为模型初始病例

    resident_grid1 = pd.read_csv('shanghai_od_202203_home2.csv')
    resident_grid = resident_grid1.groupby(['grid'])['usum'].sum().reset_index()  # 每个grid的人口，len<7355，部分grid人口为0
    grid_id = list(grid_rect.grid_id)
    for igrid in grid_id:
        mid = resident_grid[resident_grid.grid == igrid]
        if mid.shape[0] == 0:
            resident_grid = resident_grid.append({'grid': igrid, 'usum': 1}, ignore_index=True)
    resident_grid.sort_values(by=['grid'], ascending=True, inplace=True)
    resident_grid1 = resident_grid.reset_index(drop=True)
    resid_data_grid = resident_grid1.copy()
    c = 3.4/5.6
    I = np.array(grid_rect.patient)
    E = np.array([0] * len(grid_rect.patient))
    R = np.array([0] * len(grid_rect.patient))  # numpy.array可直接加减
    N = np.array(resid_data_grid.usum)
    S = N - E - R - I
    bool_I_N = np.where(I > N)
    N[bool_I_N] = I[bool_I_N]
    E = c * S * I / N  ###初始E不是0，避免开始时I降低，相当于提前反应一步
    S = N - E - R - I
    Grid_SEIR1 = {'S': list(S), 'E': list(E), 'I': list(I), 'R': list(R), 'N': list(N)}
    Grid_SEIR = pd.DataFrame(Grid_SEIR1)
    DETA_I = simulation(Grid_SEIR, 0.063, 0.2)
    # #########################画图######################
    case_data1 = pd.read_csv(f'case_data.csv')
    case_data1['time'] = pd.to_datetime(case_data1.date)
    case_data1.sort_values(by=['time'], ascending=True, inplace=True)
    start_time = '2022-03-05'
    end_time = '2022-06-01'
    case_data = case_data1.dropna(axis=0, subset=["location_bd09ll"])
    case_data2 = case_data[
        (case_data.time >= pd.to_datetime(start_time)) & (case_data.time <= pd.to_datetime(end_time))]
    case_data3 = case_data2[case_data2.whe_out != 1]
    case_data_date = case_data3.groupby(['time'])['id'].count().reset_index()
    #
    # #########模型拟合效果对比图##########
    plt.figure()
    plt.rcParams['figure.figsize']=(7.5,5.0)
    config = {
        "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 20,
        'figure.figsize':(20.0,7.0)
    }
    plt.rcParams.update(config)
    x_data=range(case_data_date.shape[0])
    # #拟合优度
    # file=open('E:\博士\研究\上海疫情\数据\shanghai_od_matrix\项目所需\\result\DETA_I_0.063_0.2.pickle','rb')
    # DETA_I=pickle.load(file)
    # file.close()
    r2 = r2_score(DETA_I[0:87], case_data_date.id)
    plt.scatter(x_data,case_data_date.id,s=40,color='#bbbbd6',label='ground-truth')#65a9d7
    plt.plot(DETA_I,linewidth=4,color='#58539f',label='RDT')#e9963e
    plt.legend()
    plt.title(f'R2：{r2}')
    print(r2)
    plt.savefig('./result/RDT.pdf',dpi=700)
    plt.show()







