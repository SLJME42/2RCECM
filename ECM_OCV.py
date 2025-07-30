import numpy as np
import math
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
#plt.style.use(['science','ieee','cjk-sc-font'])
plt.rcParams['font.family'] = 'SimHei'
# 避免负号显示为方框
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18
def KF_2ndRC(I,U,Ts):
    U0_vec=[]
    U1_vec=[]
    U2_vec=[]
    Uoc_vec=[]
    term_vec=[]
#####Initializing slow dynamic
    X_k = np.array([[2.5, 10, 200, 0.1, 0.01, 100, 0.1, 0.01]]).T
    #V_oc, tal_1, tal_2, R_p1, R_p2, R_o, V_p1, V_p2
    w_k=10**(-8)*np.eye(8, dtype=int)
    v_k=np.array([[10**(-4)]])
    P_k=10**(-2)*np.eye(8)

    for k in range(1,len(I)):

        phi1 = np.exp(-Ts / X_k[1,0])
        a1 = (phi1 / (X_k[1,0] ** 2)) * (X_k[6,0] - X_k[3,0] * I[k])
        a2 = (1 - phi1) * I[k]
        a3 = phi1

        phi2 = np.exp(-Ts / X_k[2,0])
        b1 = (phi2 / (X_k[2,0] ** 2)) * (X_k[7,0] - X_k[4,0] * I[k])
        b2 = (1 - phi2) * I[k]
        b3 = phi2

        A_k = np.array([[1, 0, 0, 0, 0,0,0,0], [0, 1, 0,0,0, 0,0,0], [0, 0, 1, 0,0,0,0, 0], [0, 0, 0, 1, 0,0,0,0],
                        [0, 0, 0, 0, 1,0,0,0], [0, 0, 0, 0, 0,1,0,0], [0, a1, 0,a2, 0,0, a3,0], [0, 0,b1, 0,b2, 0, 0,b3]])
        C_k = np.array([[1, 0, 0, 0,0,I[k], -1,-1]])


        # Prior prediction
        X_k = A_k@X_k
        P_k = A_k@P_k@A_k.T + w_k
        # Kalman gain computation
        SSS=C_k@P_k@C_k.T+v_k
        K_k = P_k@C_k.T@np.linalg.inv(SSS)
        # Posteriori update

        X_k = X_k + K_k@(U[k] - C_k@X_k)
        nn= np.eye(8, dtype=float)-K_k@C_k.reshape(1,8)
        P_k = np.dot((np.eye(8, dtype=float) - K_k@C_k.reshape(1,8)), P_k)
        term_k=np.array(C_k@X_k)[0]

        Uoc_vec.append(X_k[0])
        U0_vec.append(I[k]*X_k[5])
        U1_vec.append(X_k[6])
        U2_vec.append(X_k[7])
        term_vec.append(term_k)
    return Uoc_vec,term_vec


def RFF_KF_3U(I,U,Ts,OCV_ini):
    U0_vec=[]
    U1_vec=[]
    U2_vec=[]
    Uoc_vec=[]
    term_vec=[]
    R2_vec=[]
    tal2_vec=[]
#####Initializing fast dynamic
    c1=10/(Ts+10)#tal1/(tal1+Ts)
    c2=100+(Ts*0.1)/(Ts+10)#R0+Ts*R1/(Ts+tal1)
    c3=-1000/(Ts+10)#-R0*tal1/(Ts+tal1)
    thita_ls=np.array([[c1,c2,c3]]).T#c1,c2,c3
    P = 10**(-4)*np.eye(3)
    lamda = 1  # forgetting factor
    Ufd_k_1=1
    Ufd_k=1.1
    U1=0.1
    phi = np.array([Ufd_k_1, I[1], I[0]]).reshape(1, 3)
    #K = (P @ (phi.T)) / (lamda + phi * P @ (phi.T))
    J=0
    D=[[1,0,0],[0,0,0],[0,0,0]]
#####Initializing slow dynamic
    #w_k=10**(6)*np.eye(4)
    w_k=np.diag([10**(-8),10**(-6),10**(-8),10**(-6)])
    v_k=np.array([[10**(-4)]])
    X_k=np.array([[OCV_ini,0.01,200,0.01]]).reshape(-1,1)  #OCV,R2,tal2,U2
    P_k=10**(-4)*np.eye(4)

    time_cos = []
    for k in range(1,len(I)):
###################FRLS Fast Dynamic#############
        phi = np.array([[Ufd_k_1, I[k], I[k-1]]])
        K = (P @ (phi.T)) / (lamda + phi * P @ (phi.T))
        P = (P - K @ phi @ P) / lamda
        J=J+((Ufd_k-phi@thita_ls)**2)/(lamda + phi * P @ (phi.T))
        thita_ls = thita_ls + K @ (Ufd_k - phi @ thita_ls)
        #print(thita_ls)

        #update variables
        tal1=thita_ls[0]*Ts/(1-thita_ls[0])
        R0=(-thita_ls[2]*(Ts+tal1))/tal1
        R1=(thita_ls[1]-R0)*(Ts+tal1)/Ts
        U0=R0*I[k]
        U1=np.exp(-Ts/tal1)*U1+R1*(1-np.exp(-Ts/tal1))*I[k]
        Ufde=phi @ thita_ls
        #print([thita_ls[0],thita_ls[1]-R0])
        if isinstance(Ufd_k, (int, float)):
            Ufd_k_1=Ufd_k
        else:
            Ufd_k_1=Ufd_k[0]

        Usd_k=U[k]-Ufde
        U0_vec.append(U0)
        U1_vec.append(U1)
####################EKF Slow dynamic###############
        a=np.exp(-Ts/X_k[2][0])
        b=(1-a)*I[k]

        A_k = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 1, 0], [0,  b, 0, a]])
        C_k = np.array([[1, 0, 0, 1]])
        ckt=C_k.T
        # Prior prediction
        X_k = A_k@X_k
        P_k = A_k@P_k@A_k.T + w_k
        # Kalman gain computation
        SSS=C_k@P_k@C_k.T+v_k
        K_k = P_k@C_k.T@np.linalg.inv(SSS)
        # Posteriori update


        er = Usd_k - C_k @ X_k
        bb = K_k @ (Usd_k - C_k @ X_k)
        X_k = X_k + K_k@(Usd_k - C_k@X_k)
        #if I[k] == 0:
        #    X_k[0] = Uoc_vec[-1]
        nn= np.eye(4, dtype=float)-K_k@C_k.reshape(1,4)
        P_k = np.dot((np.eye(4, dtype=float) - K_k@C_k.reshape(1,4)), P_k)
        Usd_k=np.array(C_k@X_k)[0]

        Uoc_vec.append(X_k[0])
        R2_vec.append(X_k[1])
        tal2_vec.append(X_k[2])
        U2_vec.append(X_k[3])
        term_vec.append(Ufd_k + Usd_k)
        Ufd_k = U[k] - Usd_k
    U0_vec=np.array(U0_vec).reshape(-1,1)
    U1_vec=np.array(U1_vec).reshape(-1,1)
    U2_vec=np.array(U2_vec).reshape(-1,1)
    Uoc_vec=np.array(Uoc_vec).reshape(-1,1)
    term_vec=np.array(term_vec).reshape(-1,1)
    return Uoc_vec,term_vec

def RFF_RLS_3U(I,U,Ts,OCV_ini):
    U0_vec=[]
    U1_vec=[]
    U2_vec=[]
    Uoc_vec=[]
    term_vec=[]
    R2_vec=[]
    tal2_vec=[]
#####Initializing fast dynamic
    c1=10/(Ts+10)#tal1/(tal1+Ts)
    c2=50+(Ts*0.1)/(Ts+10)#R0+Ts*R1/(Ts+tal1)
    c3=-500/(Ts+10)#-R0*tal1/(Ts+tal1)
    thita_ls=np.array([[c1,c2,c3]]).T#c1,c2,c3
    P = 10**(-4)*np.eye(3)
    lamda = 1  # forgetting factor
    Ufd_k_1=1
    Ufd_k=1.1
    U1=0.1
    phi = np.array([Ufd_k_1, I[1], I[0]]).reshape(1, 3)
    #K = (P @ (phi.T)) / (lamda + phi * P @ (phi.T))
    J=0
    D=[[1,0,0],[0,0,0],[0,0,0]]
#####Initializing slow dynamic
    Usd_k_1=OCV_ini+0.01
    d1=200/(Ts+200)#tal2/(tal2+Ts)
    d2=200*0.01/(Ts+200)#tal2*R2/(tal2+Ts)
    d3=OCV_ini-d1*(OCV_ini-0.01)
    Uoc=OCV_ini
    thita_slow=np.array([[d1,d2,d3]]).T#c1,c2,c3
    P_slow=10**(-4)*np.eye(3)

    time_cos = []
    for k in range(1,len(I)):
###################FRLS Fast Dynamic#############
        phi = np.array([[Ufd_k_1, I[k], I[k-1]]])
        K = (P @ (phi.T)) / (lamda + phi * P @ (phi.T))
        P = (P - K @ phi @ P) / lamda
        #J=J+((Ufd_k-phi@thita_ls)**2)/(lamda + phi * P @ (phi.T))
        thita_ls = thita_ls + K @ (Ufd_k - phi @ thita_ls)
        #print(thita_ls)

        #update variables
        tal1=thita_ls[0]*Ts/(1-thita_ls[0])
        R0=(-thita_ls[2]*(Ts+tal1))/tal1
        R1=(thita_ls[1]-R0)*(Ts+tal1)/Ts
        U0=R0*I[k]
        U1=np.exp(-Ts/tal1)*U1+R1*(1-np.exp(-Ts/tal1))*I[k]
        Ufde=phi @ thita_ls
        #print([thita_ls[0],thita_ls[1]-R0])
        if isinstance(Ufd_k, (int, float)):
            Ufd_k_1=Ufd_k
        else:
            Ufd_k_1=Ufd_k[0,0]

        Usd_k=U[k]-Ufde
        U0_vec.append(U0)
        U1_vec.append(U1)
####################RLS Slow dynamic###############
        phi_slow = np.array([[Usd_k_1, I[k], 1]])
        K_slow = (P_slow @ (phi_slow.T)) / (lamda + phi_slow * P_slow @ (phi_slow.T))
        P_slow = (P_slow - K_slow @ phi_slow @ P_slow) / lamda
        thita_slow = thita_slow + K @ (Usd_k - phi_slow @ thita_slow)

        #R2=thita_slow[1]/thita_slow[0]
        #tal2=thita_slow[0]*Ts/(1-thita_slow[0])
        Uoc=thita_slow[2]+thita_slow[0]*Uoc
        Uoc_vec.append(Uoc)
        Usde=phi_slow @ thita_slow
        Usd_k_1=Usd_k[0,0]

        term_vec.append(Ufde + Usde)
        Ufd_k = U[k] - Usde
    Uoc_vec=np.array(Uoc_vec).reshape(-1,1)
    term_vec=np.array(term_vec).reshape(-1,1)
    return Uoc_vec,term_vec

def loadGITT():
    path = '/'
    raw_files = glob.glob(path + "*.xlsx")
    #dataset X: U[k-1],I[K],I[K-1],SOC[k],SOC[k-1],or add the Usd[K-1], Usd[k] =OCV+U2
    #dataset Y: U[k]
    X=[]
    Y=[]
    for file in raw_files[:2]:
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names[3]
        file_read = pd.read_excel(excel_file, sheet_name = sheet_names)
        State=file_read['State'].values
        if 'CC Chg' not in State:
            continue
        StartID=np.where(State=='CC Chg')[0][0]
        EndID=np.where(State=='CC DChg')[0][-1]
        I=file_read['Current(A)'].values[StartID:EndID]
        V=file_read['Voltage(V)'].values[StartID:EndID]
        Step=file_read['Steps'].values[StartID:EndID]
        State=State[StartID:EndID]
        Uoc_vec_KF, term_vec_KF = KF_2ndRC(I, V, 30)
        Uoc_vec_RK, term_vec_RK = RFF_KF_3U(I, V, 30, 2.5)
        Uoc_vec_RLS, term_vec_RLS = RFF_RLS_3U(I, V, 30, 2.5)
        V=V[1:]
        plt.plot(30*np.arange(1000,1600),term_vec_KF[1000:1600],label='KF-KF')
        plt.plot(30*np.arange(1000,1600),term_vec_RK[1000:1600],label='FMRLS-KF')
        plt.plot(30*np.arange(1000,1600),term_vec_RLS[1000:1600],label='FMRLS-FMRLS')
        plt.plot(30*np.arange(1000,1600),V[1000:1600],'--',label='Measured $U_{t}$')
        plt.legend()
        plt.xlabel('t (s)')
        plt.ylabel('$U_{t}$ (V)')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()
        plt.show()

        plt.plot(30*np.arange(len(V)),Uoc_vec_KF,label='KF-KF')
        plt.plot(30*np.arange(len(V)),Uoc_vec_RK,label='FMRLS-KF')
        plt.plot(30*np.arange(len(V)),Uoc_vec_RLS,label='FMRLS-FMRLS')
        plt.plot(30*np.arange(len(V)),V,'--',label='Measured $U_{t}$')
        plt.legend()
        plt.xlabel('t (s)')
        plt.ylabel('$U_{oc}$ (V)')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()
        plt.show()
        print('Done!')
    return

loadGITT()