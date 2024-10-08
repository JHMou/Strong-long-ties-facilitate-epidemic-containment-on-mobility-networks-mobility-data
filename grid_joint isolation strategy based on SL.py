import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from tqdm import tqdm
import random
from collections import Counter

def fast_od_matrix(flow_data,grid):
    G = nx.from_pandas_edgelist(flow_data, "source", "target", "weight", create_using=nx.DiGraph())
    part_grid = np.array(G.nodes())
    G.add_nodes_from(np.setdiff1d(grid, part_grid), key=int)
    A = nx.to_numpy_matrix(G, weight='weight')
    return A

def roulette(select_list):
    sum_val = 1
    random_val = random.random()
    probability = 0  
    for i in range(len(select_list)):
        probability += select_list[i] / sum_val  
        if probability >= random_val:
            return i  
        else:
            continue

def SEIR_Reaction_Diffusion(Grid_SEIR, P, c, elong, r):
    #agent-based
    S_i = []
    E_i = []
    R_i = []
    P1 = P.copy()
    v1=list(Grid_SEIR.V)
    deta_I = []
    for i in range(Grid_SEIR.shape[0]):
        if Grid_SEIR.loc[i][5] == 0:  # 判断N是否为0，没有SEIR传播模型
            Grid_SEIR.iloc[i, 1] =0
            Grid_SEIR.iloc[i, 2] =0
            Grid_SEIR.iloc[i, 3] =0
            Grid_SEIR.iloc[i, 4] =0
            deta_I.append(0)
        else:
            S_i1 = Grid_SEIR.loc[i][1]
            E_i1 = Grid_SEIR.loc[i][2]
            I_i1 = Grid_SEIR.loc[i][3]
            R_i1 = Grid_SEIR.loc[i][4]
            N_i1 = Grid_SEIR.loc[i][5]
            dS_dt1 = (S_i1 - c * S_i1 * I_i1 / N_i1-v1[i]*S_i1 ) # 自身变化情况
            dE_dt1 = (c * S_i1 * I_i1 / N_i1 + (1 - elong) * E_i1)
            dI_dt1 = (elong * E_i1 + (1 - r) * I_i1)
            dR_dt1 = (r * I_i1 + R_i1+v1[i]*S_i1)
            deta_I1 = elong * E_i1
            Grid_SEIR.iloc[i,1] = dS_dt1
            Grid_SEIR.iloc[i,2] = dE_dt1
            Grid_SEIR.iloc[i,4] = dR_dt1
            deta_I.append(deta_I1)
            #反应中I的确定也是通过随机数的方式确定
            rand_I=dI_dt1-int(dI_dt1)
            random_val = random.random()
            if rand_I>=random_val:
                Grid_SEIR.iloc[i, 3] = int(dI_dt1)+1
            else:
                Grid_SEIR.iloc[i, 3] = int(dI_dt1)
    #先反应，再转移。
    #转移概率为p*D,D=1
    #SER平均移动并取整
    Grid_SEIR2=Grid_SEIR.copy()#Grid_SEIR作为原始数据
    Grid_SEIR2['S']=0
    Grid_SEIR2['E'] = 0
    Grid_SEIR2['I'] = 0
    Grid_SEIR2['R'] = 0
    Grid_SEIR2['N'] = 0
    S_j1 = np.array(Grid_SEIR.S)
    E_j1 = np.array(Grid_SEIR.E)
    R_j1 = np.array(Grid_SEIR.R)
    for i in range(Grid_SEIR.shape[0]):
        p_j_i = P1[:, i]
        S_list = p_j_i * S_j1 # 每个grid转移过来的数据
        dS_dt2 = sum(S_list)
        E_list = p_j_i *  E_j1
        dE_dt2 = sum(E_list)
        R_list = p_j_i * R_j1
        dR_dt2 = sum(R_list)
        S_i.append(dS_dt2)
        E_i.append(dE_dt2)
        R_i.append(dR_dt2)
    Grid_SEIR2['S'] = S_i
    Grid_SEIR2['E'] = E_i
    Grid_SEIR2['R'] = R_i
    #讨论I的仿真移动
    for i in range(Grid_SEIR.shape[0]):
        if Grid_SEIR.loc[i][5] == 0:
            Grid_SEIR2.iloc[i, 3] = 0
        else:#轮盘赌的方式进行模拟,转移是同时进行的，同时对7355个进行转移
            #只讨论I的仿真流动，SER考虑平均流动
            p_j_i1 = P1[i, :]  # i的出度grid,存在p=0的情况，则只选择第一个
            index_list=[]
            if sum(p_j_i1)!=0:
                if int(Grid_SEIR.loc[i][3])!=0:
                    for j in range(int(Grid_SEIR.loc[i][3])): 
                        index_list.append(roulette(p_j_i1))
                    I_Uni = Counter(index_list)
                    try:
                        for key,b in I_Uni.items():
                            Grid_SEIR2.iloc[int(key), 3]+=b
                    except:
                        print('error')
            else:
                continue
    Grid_SEIR2['N'] = np.array(Grid_SEIR2['S']) + np.array(Grid_SEIR2['E']) + np.array(Grid_SEIR2['I']) + np.array(Grid_SEIR2['R'])
    Grid_SEIR2['deta_I'] = deta_I
    return Grid_SEIR2



def simulation_long_tie(Grid_SEIR,P_long,tage,edge,long_tie):
    edge2 = edge[['grid_id', 'target', 'weight', 'distance']]
    edge = edge2.copy()
    c = 3.4/5.6
    elong = 1 / 1.2
    r = 1 / 5.6
    icount = 1
    Grid_SEIR['control_num']=[0]*Grid_SEIR.shape[0]
    Grid_SEIR_T = Grid_SEIR.copy()
    grid_index=list(Grid_SEIR_T.id)
    I = []
    #
    D_I=[]
    cut_weight=0
    v_after=0.063
    Grid_control=Grid_SEIR.copy()
    grid_threshold=3#grid中I>threshold进行隔离
    edge.rename(columns={'grid_id': 'source'}, inplace=True)
    edge1=edge[['source','target','weight','distance']]
    grid_ini_patient_num = []
    grid_involved_num=[]
    edges_distance_num = []
    weight_total=[]
    print('==================long  tie===============')
    flag_list = [0] * Grid_control.shape[0]
    while tqdm(icount < 90):
        if icount<=26:
            Grid_SEIR_T = SEIR_Reaction_Diffusion(Grid_SEIR_T, P_long, c, elong, r)  # 控制措施不能直接删除连边，应该是减少连边权重
            Grid_control = Grid_SEIR_T.copy()  # 更新control
            I.append(sum(Grid_SEIR_T.I))
            D_I.append(sum(Grid_SEIR_T.deta_I))
            print(str(icount) + 'epoch:' + str(D_I[-1]))
            icount += 1
            grid_ini_patient_num.append(0)
            edges_distance_num.append(0)
            weight_total.append(0)
            grid_involved_num.append(0)
            continue
        else:
            grid_weight_list=[]
            #保留上一时刻Grid_SEIR_T值
            # file = open(f'/home/moujianhong/shanghai_data/results/long_tie1/Grid_SEIR_T_{icount}.pickle', 'wb')
            # pickle.dump(Grid_SEIR_T, file)
            # file.close()
            edge_num_distance=0
            #控制连边#5-10之间控制连边，本身grid的V增加
            grid_isolation = Grid_control[(Grid_control.I >= grid_threshold) & (Grid_control.flag == 0)]
            print('grid_isolation:' + str(grid_isolation.shape[0]))
            control_edges=pd.DataFrame(columns=['source','target','weight','distance'])
            if (grid_isolation.shape[0]==0):
                Grid_SEIR_T = SEIR_Reaction_Diffusion(Grid_SEIR_T, P_long, c, elong, r)  # 控制措施不能直接删除连边，应该是减少连边权重
                Grid_control = Grid_SEIR_T.copy()  # 更新control
                I.append(sum(Grid_SEIR_T.I))
                D_I.append(sum(Grid_SEIR_T.deta_I))
                print(str(icount) + 'epoch:' + str(D_I[-1]))
                grid_ini_patient_num.append(0)
                edges_distance_num.append(0)
                weight_total.append(0)
                grid_involved_num.append(0)
                icount += 1
                continue
            else:
                control_dict = {}  # 存储初始grid和受牵连grid
                itarget_all = []
                if grid_isolation.shape[0]!=0:
                    for iini_patient in range(grid_isolation.shape[0]):
                        itarget = []
                        isource = grid_isolation.iloc[iini_patient][0]
                        grid_weight_list.append(isource)
                        itarget_all.append(isource)
                        itarget_df = long_tie[long_tie.source == isource]
                        edge_num_distance += itarget_df.shape[0]
                        for j in set(list(itarget_df['target'])):
                            if flag_list[grid_index.index(j)] == 0:  
                                itarget.append(j)
                        grid_weight_list.extend(itarget) 
                        Grid_SEIR_T.loc[grid_index.index(isource), 'control_num'] = len(itarget)
                        control_dict[isource] = itarget
                        itarget_all.extend(itarget)
                    for j in list(set(itarget_all)):  
                        flag_list[grid_index.index(j)] = 1
                    grid_weight_list1 = list(set(grid_weight_list))
                    # 检查grid_distance_list1中之前没有被控制过的grid
                    grid_list1 = list(Grid_SEIR_T[Grid_SEIR_T.flag != 0]['id']) 
                    grid_weight_list2 = list(set(grid_weight_list1).difference(set(grid_list1)))

                    source_isolation=list(grid_weight_list2)
                    edge_grid1=edge1[edge1.source.isin(source_isolation) | edge1.target.isin(source_isolation)]
                    edge_grid2= edge1[edge1.source.isin(source_isolation) & edge1.target.isin(source_isolation)]
                    edge_grid = pd.concat([edge_grid1, edge_grid2], ignore_index=True, verify_integrity=True, sort=True)
                    edge_grid.drop_duplicates(subset=['source', 'target'], keep=False, inplace=True)
                    Grid_SEIR_T.loc[(Grid_SEIR_T.id.isin(source_isolation)), 'V'] = v_after
                    Grid_SEIR_T.loc[(Grid_SEIR_T.id.isin(source_isolation)), 'flag']=icount#标记grid隔离的时间
                    control_edges=pd.concat([control_edges, edge_grid], ignore_index=True, verify_integrity=True, sort=True)

                # # 所有控制的edges的weight停留在source
                control_edges.drop_duplicates(subset=['source', 'target'], keep=False, inplace=True)
                weight_total.append(sum(control_edges.weight))
                edges_distance_num.append(control_edges.shape[0])
                grid_ini_patient_num.append(grid_isolation.shape[0])
                grid_involved_num.append(edge_num_distance)

                control_edges1 = control_edges.groupby(['source'])['weight'].sum().reset_index()
                source_weight = set(list(control_edges1.source))  
                self_edges0 = edge1[(edge1.source.isin(source_weight)) & (edge1.target.isin(source_weight)) & (edge1.source == edge1.target)]
                self_edges2 = pd.merge(self_edges0, control_edges1, how='inner', on='source')
                self_edges2['weight_x'] += self_edges2['weight_y']
                self_edges2.rename(columns={'weight_x': 'weight'}, inplace=True)
                control_edges.loc[:, 'weight'] = cut_weight
                control_edges['weight_flag'] = 1
                del self_edges2['weight_y']
                edge1 = pd.concat([edge1, self_edges2], ignore_index=True, verify_integrity=True, sort=True)
                edge1.drop_duplicates(subset=['source', 'target'], keep='last', inplace=True)
                edge1 = pd.concat([edge1, control_edges], ignore_index=True, verify_integrity=True, sort=True)
                edge1.drop_duplicates(subset=['source', 'target'], keep='last', inplace=True)

                edges_weight1 = edge1.reset_index(drop=True)
                edges_weight = edges_weight1[['source', 'target', 'weight', 'distance']]
                A = fast_od_matrix(edges_weight, grid_list_new)
                P_long = normalize(np.asarray(A), axis=1, norm='l1')
                source2 = list(grid_isolation.id)
                Grid_SEIR_T.loc[(Grid_SEIR_T.id.isin(source2)), 'V'] = v_after
                Grid_SEIR_T = SEIR_Reaction_Diffusion(Grid_SEIR_T, P_long, c, elong, r)  
                Grid_control = Grid_SEIR_T.copy()  # 更新control

            I.append(sum(Grid_SEIR_T.I))
            D_I.append(sum(Grid_SEIR_T.deta_I))
            print(str(icount) + 'epoch:' + str(D_I[-1]))
            icount += 1

    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data_20240102/Real_control_{tage}.pickle', 'wb')
    pickle.dump(Grid_SEIR_T, file)
    file.close()

    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data_20240102/results/I_long_{tage}.pickle', 'wb')
    pickle.dump(I, file)
    file.close()
    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data_20240102/results/DETA_I_long_{tage}.pickle', 'wb')
    pickle.dump(D_I, file)
    file.close()
    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data_20240102/results/Weight_Cost_long_{tage}.pickle', 'wb')
    pickle.dump(weight_total, file)
    file.close()
    f=f'/data3/lvxin/moujianhong/shanghai_data/data_20240102/results/Grid_SEIR_T_long_{tage}.csv'
    Grid_SEIR_T.to_csv(f, index=False, header=True)
    EDGE = {'Grid_Cost_initial': grid_ini_patient_num,'Grid_Cost_involved':grid_involved_num, 'Edges_Cost': edges_distance_num,'Weight_Cost':weight_total}
    data_edge = pd.DataFrame(EDGE)
    file_name = f'/data3/lvxin/moujianhong/shanghai_data/data_20240102/results/Cost_long_{tage}.csv'
    data_edge.to_csv(file_name, index=False, header=True)







if __name__=='__main__':
    file = open('/data3/lvxin/moujianhong/shanghai_data/data/patient1.pickle', 'rb')
    patient0 = pickle.load(file)
    file.close()
    weight_threshold=28
    edges0 = patient0[0]  
    long_tie = pd.read_csv(f'/data3/lvxin/moujianhong/shanghai_data/data/stage1.csv')
    long_tie1=long_tie[long_tie.weight>weight_threshold]
    long_tie=long_tie1.copy()

    grid_list_new1 = list(long_tie['source'])
    grid_list_new2 = list(long_tie['target'])
    grid_list_new1.extend(grid_list_new2)
    grid_list_new = list(set(grid_list_new1))#长程边涉及的grid
    edges_new1 = edges0[edges0.grid_id.isin(grid_list_new)]
    edges_new2 = edges_new1[edges_new1.target.isin(grid_list_new)]
    edges_new = edges_new2.reset_index(drop=True)
    edges_distance=edges_new.copy()
    edges_weight=edges_new.copy()
    edges_long=edges_new.copy()

    edges_new.rename(columns={'grid_id': 'source'}, inplace=True)
    edges_new1 = edges_new[['source', 'target', 'weight', 'distance']]
    A = fast_od_matrix(edges_new1, grid_list_new)
    P_before = normalize(np.asarray(A), axis=1, norm='l1')
    case_data1 = pd.read_csv(f'/data3/lvxin/moujianhong/shanghai_data/data/case_data.csv')
    grid_rect = pd.read_csv(r'/data3/lvxin/moujianhong/shanghai_data/data/shanghai_grid.csv')
    case_data = case_data1.dropna(axis=0, subset=["location_bd09ll"])  
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

    resident_grid1 = pd.read_csv(r'/data3/lvxin/moujianhong/shanghai_data/data/shanghai_od_202203_home2.csv')
    resident_grid = resident_grid1.groupby(['grid'])['usum'].sum().reset_index()  
    grid_id = list(grid_rect.grid_id)
    for igrid in grid_id:
        mid = resident_grid[resident_grid.grid == igrid]
        if mid.shape[0] == 0:
            resident_grid = resident_grid.append({'grid': igrid, 'usum': 1}, ignore_index=True)

    resident_grid.sort_values(by=['grid'], ascending=True, inplace=True)
    resident_grid1 = resident_grid.reset_index(drop=True)
    resident_grid = resident_grid1.copy()
    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data/Resident.pickle', 'wb')
    pickle.dump(resident_grid, file)
    file.close()
    c = 3.4/5.6
    I = np.array(grid_rect.patient)
    E = np.array([0] * len(grid_rect.patient))
    R = np.array([0] * len(grid_rect.patient))  
    file = open('/data3/lvxin/moujianhong/shanghai_data/data/Resident.pickle', 'rb')
    resid_data_grid = pickle.load(file)
    file.close()
    N = np.array(resid_data_grid.usum)
    S = N - E - R - I
    D = [1] * len(grid_rect.patient)
    V = [0] * len(grid_rect.patient)
	
    bool_I_N = np.where(I > N)
    N[bool_I_N] = I[bool_I_N]
    E = c * S * I / N  
    S = N - E - R - I
    Grid_SEIR1 = {'id': grid_id, 'S': list(S), 'E': list(E), 'I': list(I), 'R': list(R), 'N': list(N), 'D': D, 'V': V}
    Grid_SEIR = pd.DataFrame(Grid_SEIR1)
    Grid_SEIR1 = Grid_SEIR[Grid_SEIR.id.isin(grid_list_new)]

    #控制措施从0305开始
    resident_grid.rename(columns={'grid': 'id'}, inplace=True)
    Grid_SEIR_long=pd.merge(Grid_SEIR1,resident_grid,on=['id'],how='inner')#
    Grid_SEIR_long['flag']=0

    Grid_SEIR_distance=Grid_SEIR_long.copy()
    Grid_SEIR_weight=Grid_SEIR_long.copy()
    print('initial:'+str(sum(Grid_SEIR_long.I)))

    itertor1=1
    for i in range(itertor1):
        Grid_SEIR_long1=Grid_SEIR_long.copy()
        P_before1=P_before.copy()
        edges_long1=edges_long.copy()
        long_tie1=long_tie.copy()
        simulation_long_tie(Grid_SEIR_long1, P_before1,str(i)+'_28_10', edges_long1,long_tie1)











