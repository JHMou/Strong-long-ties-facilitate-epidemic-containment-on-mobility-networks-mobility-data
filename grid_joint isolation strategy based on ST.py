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
    G1 = nx.DiGraph()
    node_list1 = list(flow_data.source)
    node_list1.extend(list(flow_data.target))
    node_list = list(set(node_list1))
    node_list.sort()
    my_dict = {}
    for index, item in enumerate(node_list):
        my_dict[item] = index
    EDGE = {'grid_id': node_list, 'id': list(range(len(node_list)))}
    edge_rank = pd.DataFrame(EDGE)
    edge_rank.rename(columns={'grid_id': 'source'}, inplace=True)
    flow_data1 = pd.merge(flow_data, edge_rank, how='inner', on='source')
    flow_data1.rename(columns={'id': 'source_id'}, inplace=True)
    edge_rank.rename(columns={'source': 'target'}, inplace=True)
    flow_data2 = pd.merge(flow_data1, edge_rank, how='inner', on='target')
    flow_data2.rename(columns={'id': 'target_id'}, inplace=True)
    G1.add_nodes_from(list(range(len(node_list))))
    edgelist = flow_data2[['source_id', 'target_id', 'weight']].values.tolist()
    G1.add_weighted_edges_from(edgelist)
    A = nx.to_numpy_matrix(G1, weight='weight')
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
    S_i = []
    E_i = []
    R_i = []
    P1 = P.copy()
    v1=list(Grid_SEIR.V)
    deta_I = []
    for i in range(Grid_SEIR.shape[0]):
        if Grid_SEIR.loc[i][5] == 0:  
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
            dS_dt1 = (S_i1 - c * S_i1 * I_i1 / N_i1-v1[i]*S_i1 ) 
            dE_dt1 = (c * S_i1 * I_i1 / N_i1 + (1 - elong) * E_i1)
            dI_dt1 = (elong * E_i1 + (1 - r) * I_i1)
            dR_dt1 = (r * I_i1 + R_i1+v1[i]*S_i1)
            deta_I1 = elong * E_i1
            Grid_SEIR.iloc[i,1] = dS_dt1
            Grid_SEIR.iloc[i,2] = dE_dt1
            Grid_SEIR.iloc[i,4] = dR_dt1
            deta_I.append(deta_I1)
            rand_I=dI_dt1-int(dI_dt1)
            random_val = random.random()
            if rand_I>=random_val:
                Grid_SEIR.iloc[i, 3] = int(dI_dt1)+1
            else:
                Grid_SEIR.iloc[i, 3] = int(dI_dt1)
    Grid_SEIR2=Grid_SEIR.copy()
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
        S_list = p_j_i * S_j1 
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
    for i in range(Grid_SEIR.shape[0]):
        if Grid_SEIR.loc[i][5] == 0:
            Grid_SEIR2.iloc[i, 3] = 0
        else:
            p_j_i1 = P1[i, :]  
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

def simulation_weight(Grid_SEIR,P_long,tage,edge1,long_tie):
    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data_20240306/control_process_t_23/Real_control_{tage}.pickle', 'rb')
    Real_control1 = pickle.load(file)
    file.close()
    Real_control=Real_control1[['id','control_num']]

    edge2 = edge1[['grid_id', 'target', 'weight', 'distance']]
    edge2.rename(columns={'grid_id': 'source'}, inplace=True)
    edge1 = edge2.copy()
    c = 3.4/5.6
    elong = 1 / 1.2
    r = 1 / 5.6
    icount = 1
    Grid_SEIR_T = Grid_SEIR.copy()
    grid_index = list(Grid_SEIR_T.id)
    Real_control21 = {'id': grid_index}
    Real_control2 = pd.DataFrame(Real_control21)
    Real_control3 = pd.merge(Real_control2, Real_control, how='left', on='id')
    Real_control = Real_control3.fillna(0)
    I = []
    D_I=[]
    cut_weight=0
    v_after = 0.063
    Grid_control=Grid_SEIR.copy()
    grid_threshold = 3  
    grid_ini_patient_num = []
    edges_distance_num = []
    edge1['weight_flag'] = [1] * edge1.shape[0]
    print('==================weight===============')
    flag_list = [0] * Grid_control.shape[0]
    weight_total=[]
    grid_involved_num=[]
    grid_total_num = []
    while tqdm(icount < 90):
        if icount<=26:
            Grid_SEIR_T = SEIR_Reaction_Diffusion(Grid_SEIR_T, P_long, c, elong, r) 
            Grid_control = Grid_SEIR_T.copy() 
            edges_distance_num.append(0)
            I.append(sum(Grid_SEIR_T.I))
            D_I.append(sum(Grid_SEIR_T.deta_I))
            print(str(icount) + 'epoch:' + str(D_I[-1]))
            icount += 1
            grid_ini_patient_num.append(0)
            weight_total.append(0)
            grid_involved_num.append(0)
            grid_total_num.append(0)
            continue
        else:
            grid_weight_list = []
            edge_num_distance = 0
            grid_isolation = Grid_control[(Grid_control.I >= grid_threshold) & (Grid_control.flag == 0)]
            print('grid_isolation:' + str(grid_isolation.shape[0]))
            control_edges = pd.DataFrame(columns=['source', 'target', 'weight', 'distance'])  
            if (grid_isolation.shape[0]==0):
                Grid_SEIR_T = SEIR_Reaction_Diffusion(Grid_SEIR_T, P_long, c, elong, r) 
                Grid_control = Grid_SEIR_T.copy()  
                edges_distance_num.append(0)
                grid_ini_patient_num.append(0)
                weight_total.append(0)
                grid_involved_num.append(0)
                grid_total_num.append(0)
                I.append(sum(Grid_SEIR_T.I))
                D_I.append(sum(Grid_SEIR_T.deta_I))
                print(str(icount) + 'epoch:' + str(D_I[-1]))
                icount += 1
                continue
            else:
                control_dict = {} 
                itarget_all = []
                if grid_isolation.shape[0]!=0:
                    grid_isolation_id = list(grid_isolation.id)
                    edge1_noego1 = edge1[edge1.distance != 0]  
                    ini_edges_isolation = edge1_noego1[(edge1_noego1.source.isin(grid_isolation_id))] 
                    ini_edges_isolation['rank'] = ini_edges_isolation.groupby(['source'])['weight'].rank(ascending=False, method='first')
                    itarget = []
                    mid_itarget = []
                    itarget_all.extend(grid_isolation_id)
                    grid_weight_list.extend(grid_isolation_id)
                    for source1 in grid_isolation_id:
                        num_delet = Real_control.loc[grid_index.index(source1), 'control_num']
                        if num_delet == 0:
                            control_dict[source1] = []
                            continue
                        else:
                            edge_num_distance += num_delet
                            ini_target = ini_edges_isolation[ini_edges_isolation.source == source1]
                            ini_target.rename(columns={'target': 'id'}, inplace=True)
                            ini_target.sort_values(by=['rank'], ascending=True, inplace=True, ignore_index=True)
                            grid2 = ini_target.reset_index(drop=True)
                            for j in list(grid2['id']):
                                if flag_list[grid_index.index(j)] == 0: 
                                    itarget.append(j)
                            if len(itarget) == 0:
                                control_dict[source1] = []
                                continue
                            if (0 < len(itarget)) & (len(itarget) < num_delet):
                                mid_itarget = itarget
                                grid_weight_list.extend(mid_itarget)
                                control_dict[source1] = mid_itarget
                            if len(itarget) >= num_delet:
                                mid_itarget = itarget[0:int(num_delet)]
                                grid_weight_list.extend(mid_itarget)
                                control_dict[source1] = mid_itarget
                            itarget_all.extend(mid_itarget)
                    for j in list(set(itarget_all)):  
                        flag_list[grid_index.index(j)] = 1
                    file = open(  f'/data3/lvxin/moujianhong/shanghai_data/data_20240306/control_process_t_23/strong_tie/Control_dict_{tage}_{icount}.pickle',  'wb')
                    pickle.dump(control_dict, file)
                    file.close()
                    file = open( f'/data3/lvxin/moujianhong/shanghai_data/data_20240306/control_process_t_23/strong_tie/Grid_SEIR_T_{tage}_{icount}.pickle',  'wb')
                    pickle.dump(Grid_SEIR_T, file)
                    file.close()
                    grid_total_num.append(len(list(set(itarget_all))))
                    grid_weight_list1 = list(set(grid_weight_list))
                    grid_list1 = list(Grid_SEIR_T[Grid_SEIR_T.flag != 0]['id'])  
                    grid_weight_list2 = list(set(grid_weight_list1).difference(set(grid_list1)))
                    source_isolation=list(grid_weight_list2)
                    edge_grid1=edge1[edge1.source.isin(source_isolation) | edge1.target.isin(source_isolation)]
                    edge_grid2= edge1[edge1.source.isin(source_isolation) & edge1.target.isin(source_isolation)]
                    edge_grid = pd.concat([edge_grid1, edge_grid2], ignore_index=True, verify_integrity=True, sort=True)
                    edge_grid.drop_duplicates(subset=['source', 'target'], keep=False, inplace=True)
                    Grid_SEIR_T.loc[(Grid_SEIR_T.id.isin(source_isolation)), 'V'] = v_after
                    Grid_SEIR_T.loc[(Grid_SEIR_T.id.isin(source_isolation)), 'flag']=-1
                    control_edges=pd.concat([control_edges, edge_grid], ignore_index=True, verify_integrity=True, sort=True)
                control_edges.drop_duplicates(subset=['source', 'target'], keep=False, inplace=True)
                weight_total.append(sum(control_edges.weight) + sum(edge_grid2.weight))
                control_edges_num = control_edges[control_edges.weight != 0]
                edges_distance_num.append(control_edges_num.shape[0])
                grid_ini_patient_num.append(grid_isolation.shape[0])
                grid_involved_num.append(edge_num_distance)
                control_edges1 = control_edges.groupby(['source'])['weight'].sum().reset_index()
                source_weight = set(list(control_edges1.source))  
                self_edges0 = edge1[(edge1.source.isin(source_weight)) & (edge1.target.isin(source_weight)) & ( edge1.source == edge1.target)]
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
                Grid_control = Grid_SEIR_T.copy()
            I.append(sum(Grid_SEIR_T.I))
            D_I.append(sum(Grid_SEIR_T.deta_I))
            print(str(icount) + 'epoch:' + str(D_I[-1]))
            icount += 1
    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data_20240306/control_process_t_23/strong_tie/I_weight_{tage}.pickle', 'wb')
    pickle.dump(I, file)
    file.close()
    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data_20240306/control_process_t_23/strong_tie/DETA_I_weight_{tage}.pickle', 'wb')
    pickle.dump(D_I, file)
    file.close()
    file = open(f'/data3/lvxin/moujianhong/shanghai_data/data_20240306/control_process_t_23/strong_tie/Weight_Cost_weight_{tage}.pickle', 'wb')
    pickle.dump(weight_total, file)
    file.close()
    f=f'/data3/lvxin/moujianhong/shanghai_data/data_20240306/control_process_t_23/strong_tie/Grid_SEIR_T_weight_{tage}.csv'
    Grid_SEIR_T.to_csv(f, index=False, header=True)
    EDGE = {'Grid_Cost_initial': grid_ini_patient_num,'Grid_Cost_involved':grid_involved_num,'Grid_Cost_total':grid_total_num, 'Edges_Cost': edges_distance_num,'Weight_Cost':weight_total}
    data_edge = pd.DataFrame(EDGE)
    file_name = f'/data3/lvxin/moujianhong/shanghai_data/data_20240306/control_process_t_23/strong_tie/Cost_weight_{tage}.csv'
    data_edge.to_csv(file_name, index=False, header=True)


if __name__=='__main__':
    file = open('/data3/lvxin/moujianhong/shanghai_data/data/patient1.pickle', 'rb')
    patient0 = pickle.load(file)
    file.close()
    edges0 = patient0[0] 
    weight_threshold=28
    long_tie = pd.read_csv(f'./strong long ties/strong_long_ties_stage1.csv')
    long_tie1 = long_tie[long_tie.weight > weight_threshold]
    long_tie = long_tie1.copy()
    grid_list_new1 = list(long_tie['source'])
    grid_list_new2 = list(long_tie['target'])
    grid_list_new1.extend(grid_list_new2)
    grid_list_new = list(set(grid_list_new1))
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
    grid_rect = pd.read_csv('input_of_control.csv')
    resident_grid1 = pd.read_csv(r'./shanghai_od_202203_home2.csv')
    resident_grid = resident_grid1.groupby(['grid'])['usum'].sum().reset_index()  
    grid_id = list(grid_rect.grid_id)
    for igrid in grid_id:
        mid = resident_grid[resident_grid.grid == igrid]
        if mid.shape[0] == 0:
            resident_grid = resident_grid.append({'grid': igrid, 'usum': 1}, ignore_index=True)
    resident_grid.sort_values(by=['grid'], ascending=True, inplace=True)
    resident_grid1 = resident_grid.reset_index(drop=True)
    resident_grid = resident_grid1.copy()
    file = open(f'Resident.pickle', 'wb')
    pickle.dump(resident_grid, file)
    file.close()
    c = 3.4/5.6
    I = np.array(grid_rect.patient)
    E = np.array([0] * len(grid_rect.patient))
    R = np.array([0] * len(grid_rect.patient)) 
    file = open('Resident.pickle', 'rb')
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
    resident_grid.rename(columns={'grid': 'id'}, inplace=True)
    Grid_SEIR_long=pd.merge(Grid_SEIR1,resident_grid,on=['id'],how='inner')
    Grid_SEIR_long['flag']=0
    Grid_SEIR_distance=Grid_SEIR_long.copy()
    Grid_SEIR_weight=Grid_SEIR_long.copy()
    print('initial:'+str(sum(Grid_SEIR_long.I)))
    itertor1 = 1
    for i in range(itertor1):
        Grid_SEIR_weight1 = Grid_SEIR_weight.copy()
        P_before1 = P_before.copy()
        edges_weight1 = edges_weight.copy()
        long_tie1 = long_tie.copy()
        simulation_weight(Grid_SEIR_weight1,P_before1,str(i)+'_28_10',edges_weight1,long_tie1)










