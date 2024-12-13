import numpy as np 

n_states = 32

n_actions = 4


action_dict = {'straight':0, 'left':1,'right':2,'stay':3}
P = np.zeros((n_actions,n_states,n_states))

# straight
P[action_dict['straight']][0][12] = 1.0
P[action_dict['straight']][1][0] = 1.0
P[action_dict['straight']][2][14] = 1.0
P[action_dict['straight']][3][2] = 1.0
P[action_dict['straight']][4][8] = 1.0
P[action_dict['straight']][5][4] = 1.0
P[action_dict['straight']][6][5] = 1.0
P[action_dict['straight']][7][6] = 1.0
P[action_dict['straight']][8][9] = 1.0
P[action_dict['straight']][9][10] = 1.0
P[action_dict['straight']][10][11] = 1.0
P[action_dict['straight']][11][7] = 1.0
P[action_dict['straight']][12][16] = 1.0
P[action_dict['straight']][13][1] = 1.0
P[action_dict['straight']][14][18] = 1.0
P[action_dict['straight']][15][3] = 1.0
P[action_dict['straight']][16][28] = 1.0
P[action_dict['straight']][17][13] = 1.0
P[action_dict['straight']][18][30] = 1.0
P[action_dict['straight']][19][15] = 1.0
P[action_dict['straight']][20][24] = 1.0
P[action_dict['straight']][21][20] = 1.0
P[action_dict['straight']][22][21] = 1.0
P[action_dict['straight']][23][22] = 1.0
P[action_dict['straight']][24][25] = 1.0
P[action_dict['straight']][25][26] = 1.0
P[action_dict['straight']][26][27] = 1.0
P[action_dict['straight']][27][23] = 1.0
P[action_dict['straight']][28][29] = 1.0
P[action_dict['straight']][29][17] = 1.0
P[action_dict['straight']][30][31] = 1.0
P[action_dict['straight']][31][19] = 1.0

# left
# left on highway is same as straight
P[action_dict['left']][0][9] = 1.0
P[action_dict['left']][1][0] = 1.0
P[action_dict['left']][2][11] = 1.0
P[action_dict['left']][3][2] = 1.0
P[action_dict['left']][4][8] = 1.0
P[action_dict['left']][5][12] = 1.0
P[action_dict['left']][6][5] = 1.0
P[action_dict['left']][7][14] = 1.0
P[action_dict['left']][8][1] = 1.0
P[action_dict['left']][9][10] = 1.0
P[action_dict['left']][10][3] = 1.0
P[action_dict['left']][11][7] = 1.0
P[action_dict['left']][12][16] = 1.0
P[action_dict['left']][13][4] = 1.0
P[action_dict['left']][14][18] = 1.0
P[action_dict['left']][15][6] = 1.0
P[action_dict['left']][16][25] = 1.0
P[action_dict['left']][17][13] = 1.0
P[action_dict['left']][18][27] = 1.0
P[action_dict['left']][19][15] = 1.0
P[action_dict['left']][20][24] = 1.0
P[action_dict['left']][21][28] = 1.0
P[action_dict['left']][22][21] = 1.0
P[action_dict['left']][23][30] = 1.0
P[action_dict['left']][24][17] = 1.0
P[action_dict['left']][25][26] = 1.0
P[action_dict['left']][26][19] = 1.0
P[action_dict['left']][27][23] = 1.0
P[action_dict['left']][28][29] = 1.0
P[action_dict['left']][29][20] = 1.0
P[action_dict['left']][30][31] = 1.0
P[action_dict['left']][31][22] = 1.0

# right
# right on highway is straight
P[action_dict['right']][0][4] = 1.0
P[action_dict['right']][1][0] = 1.0
P[action_dict['right']][2][6] = 1.0
P[action_dict['right']][3][2] = 1.0
P[action_dict['right']][4][8] = 1.0
P[action_dict['right']][5][1] = 1.0
P[action_dict['right']][6][5] = 1.0
P[action_dict['right']][7][3] = 1.0
P[action_dict['right']][8][12] = 1.0
P[action_dict['right']][9][10] = 1.0
P[action_dict['right']][10][14] = 1.0
P[action_dict['right']][11][7] = 1.0
P[action_dict['right']][12][16] = 1.0
P[action_dict['right']][13][9] = 1.0
P[action_dict['right']][14][18] = 1.0
P[action_dict['right']][15][11] = 1.0
P[action_dict['right']][16][20] = 1.0
P[action_dict['right']][17][13] = 1.0
P[action_dict['right']][18][22] = 1.0
P[action_dict['right']][19][15] = 1.0
P[action_dict['right']][20][24] = 1.0
P[action_dict['right']][21][17] = 1.0
P[action_dict['right']][22][21] = 1.0
P[action_dict['right']][23][19] = 1.0
P[action_dict['right']][24][28] = 1.0
P[action_dict['right']][25][26] = 1.0
P[action_dict['right']][26][30] = 1.0
P[action_dict['right']][27][23] = 1.0
P[action_dict['right']][28][29] = 1.0
P[action_dict['right']][29][25] = 1.0
P[action_dict['right']][30][31] = 1.0
P[action_dict['right']][31][27] = 1.0

P[action_dict['stay']] = np.eye((n_states,n_states))

