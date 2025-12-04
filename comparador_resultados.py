#%%
'''Comparador de resultados ESAR '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
#%%Funciones
#%% PLOTEADOR CICLOS PROMEDIO
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    """
    Buscar recursivamente todos los archivos que coincidan con el patrón
    *ciclo_promedio*.txt en el directorio especificado y sus subdirectorios,
    y graficar sus ciclos de histéresis.

    Parameters
    ----------
    directorio : str
        Directorio donde se busca recursivamente

    Returns
    -------
    None
    """
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
    """
    Lee archivo resultados.txt devuelve los datos y metadatos

    Parameters
    ----------
    path : str
        Ruta del archivo a leer

    Returns
    -------
    meta : dict
        Diccionario con metadatos del archivo
    files : numpy.ndarray
        Nombres de los archivos procesados
    time : numpy.ndarray
        Tiempos de medición en minutos
    temperatura : numpy.ndarray
        Temperaturas de medición en ºC
    Mr : numpy.ndarray
        Remanencia en A/m
    Hc : numpy.ndarray
        Coercitividad en kA/m
    campo_max : numpy.ndarray
        Campo máximo en kA/m
    mag_max : numpy.ndarray
        Magnetización máxima en A/m
    xi_M_0 : numpy.ndarray
        Magnetización en A/m a 0 K
    frecuencia_fund : numpy.ndarray
        Frecuencia fundamental en Hz
    magnitud_fund : numpy.ndarray
        Magnitud de la frecuencia fundamental en A/m
    dphi_fem : numpy.ndarray
        Ángulo de fase de la frecuencia fundamental en rad
    SAR : numpy.ndarray
        SAR en W/g
    tau : numpy.ndarray
        Constante de tiempo de relajación en s
    N : numpy.ndarray
        Número de datos utilizados para la ajuste"""
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
    """
    Lee un archivo de texto y devuelve los datos de los ciclos y metadatos

    Parameters
    ----------
    filepath : str
        Ruta del archivo a leer

    Returns
    -------
    t : numpy.ndarray
        Tiempos de los ciclos
    H_Vs : numpy.ndarray
        Campo de los ciclos en Vs
    M_Vs : numpy.ndarray
        Magnetizacion de los ciclos en Vs
    H_kAm : numpy.ndarray
        Campo de los ciclos en kA/m
    M_Am : numpy.ndarray
        Magnetizacion de los ciclos en A/m
    metadata : dict
        Diccionario con metadatos del archivo
    """
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
#%% Clase ResultadosESAR
#%% Clase ResultadosESAR
class ResultadosESAR:
    def __init__(self, meta, files, time, temperatura, Mr, Hc, campo_max, mag_max,
                 xi_M_0, frecuencia_fund, magnitud_fund, dphi_fem, SAR, tau, N):
        self.meta = meta
        self.files = files
        self.time = time
        self.temperatura = temperatura
        self.Mr = Mr
        self.Hc = Hc
        self.campo_max = campo_max
        self.mag_max = mag_max
        self.xi_M_0 = xi_M_0
        self.frecuencia_fund = frecuencia_fund
        self.magnitud_fund = magnitud_fund
        self.dphi_fem = dphi_fem
        self.SAR = SAR
        self.tau = tau
        self.N = N
        
        # Atributos para los ciclos
        self.primer_ciclo = None
        self.ultimo_ciclo = None
        self.primer_ciclo_metadata = None
        self.ultimo_ciclo_metadata = None
        
    def cargar_ciclos_extremos(self, directorio_base):
        """Carga el primer y último ciclo desde el subdirectorio ciclos_H_M"""
        # Determinar el directorio de ciclos
        directorio_ciclos = os.path.join(directorio_base, 'ciclos_H_M')
        
        if not os.path.exists(directorio_ciclos):
            print(f"Directorio de ciclos no encontrado: {directorio_ciclos}")
            return False
        
        # Obtener nombres del primer y último archivo de datos
        if len(self.files) == 0:
            print("No hay archivos de datos registrados")
            return False
        
        # Extraer nombres base (sin extensión .txt)
        primer_archivo_base = os.path.splitext(self.files[0])[0]  # Ej: '135kHz_100dA_100Mss_bobN1LB97OH0008'
        ultimo_archivo_base = os.path.splitext(self.files[-1])[0]  # Ej: '135kHz_100dA_100Mss_bobN1LB97OH0700'
        
        # Construir nombres de archivos de ciclos
        primer_ciclo_nombre = f"{primer_archivo_base}_ciclo_H_M.txt"
        ultimo_ciclo_nombre = f"{ultimo_archivo_base}_ciclo_H_M.txt"
        
        primer_ciclo_path = os.path.join(directorio_ciclos, primer_ciclo_nombre)
        ultimo_ciclo_path = os.path.join(directorio_ciclos, ultimo_ciclo_nombre)
        
        # Cargar primer ciclo
        if os.path.exists(primer_ciclo_path):
            try:
                t1, H_Vs1, M_Vs1, H_kAm1, M_Am1, meta1 = lector_ciclos(primer_ciclo_path)
                self.primer_ciclo = (H_kAm1, M_Am1)
                self.primer_ciclo_metadata = meta1
                print(f"  ✓ Primer ciclo cargado: {primer_ciclo_nombre}")
                print(f"    Temperatura: {meta1['Temperatura']}°C")
            except Exception as e:
                print(f"  ✗ Error cargando primer ciclo: {e}")
        else:
            print(f"  ✗ No se encontró primer ciclo: {primer_ciclo_path}")
        
        # Cargar último ciclo
        if os.path.exists(ultimo_ciclo_path):
            try:
                t2, H_Vs2, M_Vs2, H_kAm2, M_Am2, meta2 = lector_ciclos(ultimo_ciclo_path)
                self.ultimo_ciclo = (H_kAm2, M_Am2)
                self.ultimo_ciclo_metadata = meta2
                print(f"  ✓ Último ciclo cargado: {ultimo_ciclo_nombre}")
                print(f"    Temperatura: {meta2['Temperatura']}°C")
            except Exception as e:
                print(f"  ✗ Error cargando último ciclo: {e}")
        else:
            print(f"  ✗ No se encontró último ciclo: {ultimo_ciclo_path}")
        
        return True

#%% Implementación
dir_raiz_ejemplo = "LB97OH"

dir_RT = glob(os.path.join(dir_raiz_ejemplo, '**', '*_RT','**','Analisis*'), recursive=True)
dir_no_RT = glob(os.path.join(dir_raiz_ejemplo, '**', 'Analisis*'), recursive=True)
dir_no_RT = [d for d in dir_no_RT if '_RT' not in d and d not in dir_RT]
print(f"Analisis en RT: {len(dir_RT)}")
print(f"Analisis NO en RT: {len(dir_no_RT)}")

# Lista para almacenar todos los resultados
resultados_RT = []

for d in dir_RT:
    print(f"\nProcesando directorio RT: {d}")
    
    # Buscar archivo resultados.txt en este directorio
    archivos_resultados = glob(os.path.join(d, '*resultados.txt'))
    
    if not archivos_resultados:
        print(f"  No se encontró archivo resultados.txt en {d}")
        continue
    
    for archivo_resultados in archivos_resultados:
        print(f"  Procesando archivo: {os.path.basename(archivo_resultados)}")
        
        try:
            # Usar tu función lector_resultados para leer el archivo
            meta, files, time, temperatura, Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund, dphi_fem, SAR, tau, N = lector_resultados(archivo_resultados)
            
            # Crear instancia de la clase ResultadosESAR
            resultado = ResultadosESAR(
                meta=meta,
                files=files,
                time=time-time[0],
                temperatura=temperatura,
                Mr=Mr,
                Hc=Hc,
                campo_max=campo_max,
                mag_max=mag_max,
                xi_M_0=xi_M_0,
                frecuencia_fund=frecuencia_fund,
                magnitud_fund=magnitud_fund,
                dphi_fem=dphi_fem,
                SAR=SAR,
                tau=tau,
                N=N)
            
            # Guardar en la lista
            resultados_RT.append(resultado)
            
            # Mostrar información básica
            print(f"    ✓ Archivo procesado exitosamente")
            print(f"    - Nombre del archivo: {meta.get('Archivo_datos', 'N/A')}")
            print(f"    - Número de mediciones: {len(files)}")
            print(f"    - Rango de temperatura: {resultado.temperatura.min():.1f}°C - {resultado.temperatura.max():.1f}°C")
            print(f"    - Hc promedio: {resultado.Hc.mean():.2f} kA/m")
            print(f"    - SAR promedio: {resultado.SAR.mean():.2f} W/g")
            
        except Exception as e:
            print(f"  ✗ Error procesando {archivo_resultados}: {str(e)}")
            import traceback
            traceback.print_exc()

print(f"\nResumen final:")
print(f"Directorios RT procesados: {len(dir_RT)}")
print(f"Resultados ESAR cargados: {len(resultados_RT)}")


plt.plot(resultados_RT[0].time,resultados_RT[0].temperatura,'o-' ,label='RT')
#%%
# Lista para almacenar todos los resultados NO RT
resultados_no_RT = []

for d in dir_no_RT:
    print(f"\nProcesando directorio NO RT: {d}")
    # Buscar archivo resultados.txt en este directorio
    archivos_resultados = glob(os.path.join(d, '*resultados.txt'))
    
    if not archivos_resultados:
        print(f"  No se encontró archivo resultados.txt en {d}")
        continue
    
    for archivo_resultados in archivos_resultados:
        print(f"  Procesando archivo: {os.path.basename(archivo_resultados)}")
        try:
            # Usar tu función lector_resultados para leer el archivo
            meta, files, time, temperatura, Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund, dphi_fem, SAR, tau, N = lector_resultados(archivo_resultados)
            
            # Crear instancia de la clase ResultadosESAR
            resultado = ResultadosESAR(
                meta=meta,
                files=files,
                time=time-time[0],
                temperatura=temperatura,
                Mr=Mr,
                Hc=Hc,
                campo_max=campo_max,
                mag_max=mag_max,
                xi_M_0=xi_M_0,
                frecuencia_fund=frecuencia_fund,
                magnitud_fund=magnitud_fund,
                dphi_fem=dphi_fem,
                SAR=SAR,
                tau=tau,
                N=N)
            
            # Guardar en la lista
            resultados_no_RT.append(resultado)
            
            # Mostrar información básica
            print(f"    ✓ Archivo procesado exitosamente")
            print(f"    - Nombre del archivo: {meta.get('Archivo_datos', 'N/A')}")
            print(f"    - Número de mediciones: {len(files)}")
            print(f"    - Rango de temperatura: {resultado.temperatura.min():.1f}°C - {resultado.temperatura.max():.1f}°C")
            print(f"    - Hc promedio: {resultado.Hc.mean():.2f} kA/m")
            print(f"    - SAR promedio: {resultado.SAR.mean():.2f} W/g")
            
        except Exception as e:
            print(f"  ✗ Error procesando {archivo_resultados}: {str(e)}")
            import traceback
            traceback.print_exc()

print(f"\nResumen final:")
print(f"Resultados RT cargados: {len(resultados_RT)}")
print(f"Resultados NO RT cargados: {len(resultados_no_RT)}")
#%% Templog
fig,ax = plt.subplots(figsize=(8,6),constrained_layout=True)
for res in resultados_no_RT:
    ax.plot(res.time,res.temperatura,label='No RT')
ax.grid()
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Temperatura (°C)')
#% SAR
fig2,ax2 = plt.subplots(figsize=(8,5),constrained_layout=True)
for res in resultados_no_RT:
    ax2.plot(res.temperatura,res.SAR,'o-',label='No RT')
for res in resultados_RT:
    ax2.plot(res.temperatura,res.SAR,'o-',label='RT')
ax2.grid()
ax2.set_xlabel('Temperatura (°C)')
ax2.set_ylabel('SAR (W/g)')
ax2.legend()
ax2.set_title('Comparación SAR RT vs No RT')
#% Hc
fig3,ax3 = plt.subplots(figsize=(8,5),constrained_layout=True)
for res in resultados_no_RT:
    ax3.plot(res.temperatura,res.Hc,'o-',label='No RT')
for res in resultados_RT:
    ax3.plot(res.temperatura,res.Hc,'o-',label='RT')
ax3.grid()
ax3.set_xlabel('Temperatura (°C)')
ax3.set_ylabel('Hc (kA/m)')
ax3.legend()
ax3.set_title('Comparación Hc RT vs No RT')
# %Tau    
fig4,ax4 = plt.subplots(figsize=(8,5),constrained_layout=True)
for res in resultados_no_RT:
    ax4.plot(res.temperatura,res.tau,'o-',label='No RT')
for res in resultados_RT:
    ax4.plot(res.temperatura,res.tau,'o-',label='RT')
ax4.grid()
ax4.set_xlabel('Temperatura (°C)')
ax4.set_ylabel('Tau (s)')
ax4.legend()
ax4.set_title('Comparación Tau RT vs No RT')
#% Mr  
fig5,ax5 = plt.subplots(figsize=(8,5),constrained_layout=True)
for res in resultados_no_RT:
    ax5.plot(res.temperatura,res.Mr,'o-',label='No RT')
for res in resultados_RT:
    ax5.plot(res.temperatura,res.Mr,'o-',label='RT')
ax5.grid()
ax5.set_xlabel('Temperatura (°C)')
ax5.set_ylabel('Mr (A/m)')
ax5.legend()
ax5.set_title('Comparación Remanaencia RT vs No RT')
#%%


# %%
