#%%Comparador resultados de medidas ESAR sintesis de Elisa 
# Medidas a 300 kHz y 57 kA/m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
#%% Funciones
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    archivos = glob(os.path.join(directorio, '**', '*ciclo_promedio*.txt'), recursive=True)
    archivos.sort()
    if not archivos:
        print(f"No se encontraron archivos '*ciclo_promedio.txt' en {directorio} o sus subdirectorios")
        return
    fig,ax=plt.subplots(figsize=(8, 6),constrained_layout=True)
    for archivo in archivos:
        try:
            # Leer los metadatos (primeras líneas que comienzan con #)
            metadatos = {}
            with open(archivo, 'r') as f:
                for linea in f:
                    if not linea.startswith('#'):
                        break
                    if '=' in linea:
                        clave, valor = linea.split('=', 1)
                        clave = clave.replace('#', '').strip()
                        metadatos[clave] = valor.strip()

            # Leer los datos numéricos
            datos = np.loadtxt(archivo, skiprows=9)  # Saltar las 8 líneas de encabezado/metadatos

            tiempo = datos[:, 0]
            campo = datos[:, 3]  # Campo en kA/m
            magnetizacion = datos[:, 4]  # Magnetización en A/m

            # Crear etiqueta para la leyenda
            nombre_base = os.path.split(archivo)[-1].split('_')[1]
            #os.path.basename(os.path.dirname(archivo))  # Nombre del subdirectorio
            etiqueta = f"{nombre_base}"

            # Graficar

            ax.plot(campo, magnetizacion, label=etiqueta)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue

    plt.xlabel('H (kA/m)')
    plt.ylabel('M (A/m)')
    plt.title(f'Comparación de ciclos de histéresis {os.path.split(directorio)[-1]}')
    plt.grid(True)
    plt.legend()  # Leyenda fuera del gráfico
    plt.savefig('comparativa_ciclos_'+os.path.split(directorio)[-1]+'.png',dpi=300)
    plt.show()
#%% LECTOR RESULTADOS
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

#%% Localizo ciclos y resultados
ciclos_H2O=glob(('LB97OH/251112_112723_RT/**/*ciclo_promedio*'),recursive=True)
ciclos_H2O.sort()
ciclos_CPA=glob(('LB97OH+VS55/251112_134613_RT/**/*ciclo_promedio*'),recursive=True)
ciclos_CPA.sort()
 
labels=['LB97OH en H2O','LB97Oh en VS55']
#%% comparo los ciclos promedio
# plot_ciclos_promedio('300_57-20/F1_en_H')
# plot_ciclos_promedio('300_57-20/7Zgde_en_H')
# plot_ciclos_promedio('300_57-20/C1_en_H')
# plot_ciclos_promedio('300_57-20/C2_en_H')

#%% SAR y tau 

res_F1 = glob('300_57-20/F1_en_H/**/*resultados*',recursive=True)
res_F1.sort()
res_F2 = glob('300_57-20/7Zgde_en_H/**/*resultados*',recursive=True)
res_F2.sort()
res_C1 = glob('300_57-20/C1_en_H/**/*resultados*',recursive=True)
res_C1.sort()
res_C2 = glob('300_57-20/C2_en_H/**/*resultados*',recursive=True)
res_C2.sort()
res=[res_F1,res_F2,res_C1,res_C2]
for r in res:
    r.sort()

#%% Extraigo datos de las tablas de resultados       
SAR_F1,err_SAR_F1,tau_F1,err_tau_F1,hc_F1, err_hc_F1 = [],[],[],[],[],[]
SAR_F2,err_SAR_F2,tau_F2,err_tau_F2,hc_F2, err_hc_F2 = [],[],[],[],[],[]
SAR_C1,err_SAR_C1,tau_C1,err_tau_C1,hc_C1, err_hc_C1 = [],[],[],[],[],[]
SAR_C2,err_SAR_C2,tau_C2,err_tau_C2,hc_C2, err_hc_C2 = [],[],[],[],[],[]

for A in res_F1:
    meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_=lector_resultados(A)
    SAR_F1.append(meta['SAR_W/g'].n)
    err_SAR_F1.append(meta['SAR_W/g'].s)
    tau_F1.append(meta['tau_ns'].n)
    err_tau_F1.append(meta['tau_ns'].s)
    hc_F1.append(meta['Hc_kA/m'].n)
    err_hc_F1.append(meta['Hc_kA/m'].s)   

for B in res_F2:
    meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_=lector_resultados(B)
    SAR_F2.append(meta['SAR_W/g'].n)
    err_SAR_F2.append(meta['SAR_W/g'].s)
    tau_F2.append(meta['tau_ns'].n)
    err_tau_F2.append(meta['tau_ns'].s)
    hc_F2.append(meta['Hc_kA/m'].n)
    err_hc_F2.append(meta['Hc_kA/m'].s)

for C in res_C1:
    meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_=lector_resultados(C)    
    SAR_C1.append(meta['SAR_W/g'].n)
    err_SAR_C1.append(meta['SAR_W/g'].s)
    tau_C1.append(meta['tau_ns'].n)
    err_tau_C1.append(meta['tau_ns'].s)
    hc_C1.append(meta['Hc_kA/m'].n)
    err_hc_C1.append(meta['Hc_kA/m'].s)

for D in res_C2:
    meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_=lector_resultados(D)
    SAR_C2.append(meta['SAR_W/g'].n)
    err_SAR_C2.append(meta['SAR_W/g'].s)
    tau_C2.append(meta['tau_ns'].n)
    err_tau_C2.append(meta['tau_ns'].s)
    hc_C2.append(meta['Hc_kA/m'].n)
    err_hc_C2.append(meta['Hc_kA/m'].s) 

#%% SAR# 
fig, a = plt.subplots(nrows=1, figsize=(7,5), constrained_layout=True)
categories = ['F1', 'F2', 'C1', 'C2']
H = [57.0,47.8, 38.5, 29.2,19.9]

a.errorbar(x=H,y=SAR_F1, yerr=err_SAR_F1, fmt='.',ls='-', label='F1', capsize=5)
a.errorbar(x=H,y=SAR_F2, yerr=err_SAR_F2, fmt='.',ls='-', label='F2', capsize=5)
a.errorbar(x=H,y=SAR_C1, yerr=err_SAR_C1, fmt='.',ls='-', label='C1', capsize=5)
a.errorbar(x=H,y=SAR_C2, yerr=err_SAR_C2, fmt='.',ls='-', label='C2', capsize=5)

a.set_title('SAR vs H$_0$\n300 kHz')

a.set_xlabel('H$_0$ (kA/m)')
a.set_ylabel('SAR (W/g)')
a.legend(ncol=2, loc='best')
a.grid()

a.set_xticks(H)  
a.set_xticklabels(H)  
plt.savefig('SAR_vs_H0.png', dpi=300)
plt.show()
#%% tau
fig, b = plt.subplots(nrows=1, figsize=(7,5), constrained_layout=True)
categories = ['F1', 'F2', 'C1', 'C2']
H = [57.0,47.8, 38.5, 29.2,19.9]

b.errorbar(x=H,y=tau_F1, yerr=err_tau_F1, fmt='.',ls='-', label='F1', capsize=5)
b.errorbar(x=H,y=tau_F2, yerr=err_tau_F2, fmt='.',ls='-', label='F2', capsize=5)        
b.errorbar(x=H,y=tau_C1, yerr=err_tau_C1, fmt='.',ls='-', label='C1', capsize=5)
b.errorbar(x=H,y=tau_C2, yerr=err_tau_C2, fmt='.',ls='-', label='C2', capsize=5)

b.set_title('tau vs H$_0$\n300 kHz')

b.set_xlabel('H$_0$ (kA/m)')
b.set_ylabel(r'$\tau$ (ns)')
b.legend(ncol=2, loc='best')
b.grid()

b.set_xticks(H)  
b.set_xticklabels(H)  

plt.savefig('tau_vs_H0.png', dpi=300)
plt.show()  
#%% Hc 
fig,c = plt.subplots(nrows=1, figsize=(7,5), constrained_layout=True)
categories = ['F1', 'F2', 'C1', 'C2']        
H = [57.0,47.8, 38.5, 29.2,19.9]

c.errorbar(x=H,y=hc_F1, yerr=err_hc_F1, fmt='.',ls='-', label='F1', capsize=5)        
c.errorbar(x=H,y=hc_F2, yerr=err_hc_F2, fmt='.',ls='-', label='F2', capsize=5)
c.errorbar(x=H,y=hc_C1, yerr=err_hc_C1, fmt='.',ls='-', label='C1', capsize=5)        
c.errorbar(x=H,y=hc_C2, yerr=err_hc_C2, fmt='.',ls='-', label='C2', capsize=5)

c.set_title('Hc vs H$_0$\n300 kHz')

c.set_xlabel('H$_0$ (kA/m)')        
c.set_ylabel('H$_c$ (kA/m)')
c.legend(ncol=2, loc='best')
c.grid()

c.set_xticks(H)  
c.set_xticklabels(H)  

plt.savefig('Hc_vs_H0.png', dpi=300)
plt.show()
#%%% Comparo todos los graficos promediados  a mismo campo 
t_H2O,_,_,H_kAm_H2O,M_Am_H2O,metadata_H2O = lector_ciclos(ciclos_H2O[0])
t_CPA,_,_,H_kAm_CPA,M_Am_CPA,metadata_CPA = lector_ciclos(ciclos_CPA[0])
concentracion = [4.09,1.72] #g/l

fig,ax = plt.subplots(nrows=1, figsize=(7,5.5), constrained_layout=True)
ax.plot(H_kAm_H2O/1000,M_Am_H2O/concentracion[0],label='LB97OH en H2O - 4.09 g/l')
ax.plot(H_kAm_CPA/1000,M_Am_CPA/concentracion[1],label='LB97OH en VS55 - 1.72 g/l')
ax.set_title(f'Comparativa ciclos promedio a f = 135 kHz y H$_0$ = 38 kA/m ')
ax.set_xlabel('H (kA/m)')        
ax.set_ylabel('M/[NPM] (Am²/kg)')
ax.grid()
ax.legend(ncol=1, loc='best')

plt.savefig('Comparativa_ciclos_promedio_LB97OH_en_H2O_vs_VS55.png', dpi=300)
plt.show()
#%% Comparo ciclos finales RT con ciclos medidos a RT

ciclos_1=glob(('LB97OH/251251112_113920/**/*ciclo_promedio*'),recursive=True)
ciclos_CPA=glob(('LB97OH+VS55/251112_134613_RT/**/*ciclo_promedio*'),recursive=True)

t_1,_,_,H_kAm_1,M_Am_1,metadata_1 = lector_ciclos(ciclos_1[0])
t_2,_,_,H_kAm_2,M_Am_2,metadata_2 = lector_ciclos(ciclos_2[0])
t_3,_,_,H_kAm_3,M_Am_3,metadata_3 = lector_ciclos(ciclos_3[0])



t_CPA,_,_,H_kAm_CPA,M_Am_CPA,metadata_CPA = lector_ciclos(ciclos_CPA[0])
concentracion = [4.09,1.72] #g/l

fig,ax = plt.subplots(nrows=1, figsize=(7,5.5), constrained_layout=True)
ax.plot(H_kAm_H2O/1000,M_Am_H2O/concentracion[0],label='LB97OH en H2O - 4.09 g/l')
ax.plot(H_kAm_CPA/1000,M_Am_CPA/concentracion[1],label='LB97OH en VS55 - 1.72 g/l')
ax.set_title(f'Comparativa ciclos promedio a f = 135 kHz y H$_0$ = 38 kA/m ')
ax.set_xlabel('H (kA/m)')        
ax.set_ylabel('M/[NPM] (Am²/kg)')
ax.grid()
ax.legend(ncol=1, loc='best')

plt.savefig('Comparativa_ciclos_promedio_LB97OH_en_H2O_vs_VS55.png', dpi=300)
plt.show()



























