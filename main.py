from scipy.io.wavfile import read, write
from scipy.signal import lfilter, resample, butter, sawtooth, boxcar, get_window, stft
from matplotlib.mlab import specgram
from numpy import iinfo, arange, reshape, interp, linspace, cos, log2, sqrt, zeros, pi, append, float32, fft, roll, multiply, sinc
from matplotlib.pyplot import figure, plot, show, xlabel, ylabel, xlim, ylim, magnitude_spectrum
import numpy as np

def getFileValues(filename):
    (fm, s) = read(filename)
    s = s/iinfo(s.dtype).max
    n = len(s)
    return fm, s, n

def plotSound(x, y, figureLabel,xAxisLabel, yAxisLabel, xRange):
    figure(figureLabel)
    plot(x, y)
    xlim(xRange[0], xRange[1])
    xlabel(xAxisLabel)
    ylabel(yAxisLabel)
    show()

def plotAmpVsTime(fileName, figureLabel, plotRange):
    (fm, s, n) = getFileValues(fileName)
    dt = 1/fm
    t = arange(n)*dt*1000
    plotSound(t, s, figureLabel, 'Tiempo (ms)', '', plotRange)
    return 0

def plotFreqVSAmp(fileName, figureLabel, plotRange):
    (fm, s, n) = getFileValues(fileName)
    (S, f, tt) = specgram(s, n, fm, detrend=None, window=None, noverlap=None, pad_to=None, sides=None,
                         scale_by_freq=None, mode='magnitude')
    S = reshape(S, len(S))
    S = S/(n/4)
    plotSound(f, S, figureLabel, 'Freq (Hz)', '', plotRange)
    return 0

def plotRelativeSonority(fileName, figureLabel, plotRange):
    (fm, s, n) = getFileValues(fileName)
    (S, f, tt) = specgram(s, n, fm, detrend=None, window=None, noverlap=None, pad_to=None, sides=None,
                         scale_by_freq=None, mode='magnitude')
    S = reshape(S, len(S))
    S = S/(n/4)

    fMax = fm/2
    x = linspace(0, 5.7*log2(1+(fMax/230)), 1000)
    fc = (2**(x/5.7)-1)*230
    S = sqrt(interp(fc, f, S))
    plotSound(x, (S/max(S)), figureLabel, 'Distancia desde la base de la cóclea (mm)',
              'Sonoridad específica relativa (sones)', plotRange)

    return 0

def getDifferenceAvgMagnitude(s, n):
    def calculateD_k(k):
        d_k = 0
        for index in arange(1, n-k):
            d_k += abs(s[index]-s[index+k])
        return ((d_k)/(n-k))
    return [calculateD_k(k) for k in arange(0,n)]


def plotDifferenceAvgMagnitude(fileName, label, plotRange):
    (fm, s, n) = getFileValues(fileName)
    d = getDifferenceAvgMagnitude(s, n)
    t = arange(n)*(1/fm)*1000
    plotSound(t, d, label, 'Magnitud de diferencia (ms)', '', plotRange)
    return 0

def get2_1():
    fileList = ['a', 'e', 'i', 'o', 'u', 'm']
    subject = 'Adrian'
    plotRange = (200, 300)
    for letter in fileList:
        label = letter+' de ' + subject
        fileName = 'T1/Muestras/1/'+letter+'m1.wav'
        plotAmpVsTime(fileName, label, plotRange)


    plotRange = (0, 2000)
    for letter in fileList:
        label = letter+' de ' + subject
        fileName = 'T1/Muestras/1/'+letter+'m1.wav'
        plotFreqVSAmp(fileName, label, plotRange)


    plotRange = (0, 37)
    for letter in fileList:
        label = letter+' de ' + subject
        fileName = 'T1/Muestras/1/'+letter+'m1.wav'
        plotRelativeSonority(fileName, label, plotRange)

    plotRange = (0, 500)
    for letter in fileList:
        label = letter+' de ' + subject
        fileName = 'T1/Muestras/1/'+letter+'m1.wav'
        plotDifferenceAvgMagnitude(fileName, label, plotRange)
    return 0


def get2_1_2():
    fileList = ['a', 'e', 'i', 'o', 'u', 'm']
    subject = 'Sivana'
    plotRange = (200, 300)
    for letter in fileList:
        label = letter + ' de ' + subject
        fileName = 'T1/Muestras/1/' + letter + 'f1.wav'
        plotAmpVsTime(fileName, label, plotRange)

    plotRange = (0, 500)
    for letter in fileList:
        label = letter + ' de ' + subject
        fileName = 'T1/Muestras/2/' + letter + 'f2.wav'
        plotDifferenceAvgMagnitude(fileName, label, plotRange)
    return 0

def getGeometricMean(a):
    freqProduct = a[0]
    for index in arange(1,len(a)):
        freqProduct *= a[index]
    return freqProduct**(1/len(a))

def get2_1_3_a():
    #            [a,e,i,o,u,m]
    freqAdrian = [137, 138, 141, 135, 142, 137]
    freqSivana = [217, 215, 211, 200, 212, 213]
    print('Adrian\' Geometric Mean: '+str(getGeometricMean(freqAdrian))+'\nSivana\'s Geometric Mean: '+str(getGeometricMean(freqSivana)))
    return 0


def get2_2():
    fileList = ['f', 's']
    subject = 'Sivana'
    plotRange = (100, 400)
    for letter in fileList:
        label = letter + ' de ' + subject
        fileName = 'T1/Muestras/1/' + letter + 'f1.wav'
        plotAmpVsTime(fileName, label, plotRange)

    plotRange = (100, 3000)
    for letter in fileList:
        label = letter + ' de ' + subject
        fileName = 'T1/Muestras/1/' + letter + 'f1.wav'
        plotFreqVSAmp(fileName, label, plotRange)

    plotRange = (0, 37)
    for letter in fileList:
        label = letter + ' de ' + subject
        fileName = 'T1/Muestras/1/' + letter + 'f1.wav'
        plotRelativeSonority(fileName, label, plotRange)
    return 0

#_______________________________________________________________TAREA@2_________________________________________________

def convolucionarSonidos(voicePath, impulsePath):
    (fm1, voice, len1) = getFileValues(voicePath)
    t1 = arange(len1)*1/fm1*1000
    (fm2, impulse, len2) = getFileValues(impulsePath)
    t2 = arange(len2) * 1 / fm2 * 1000
    plotSound(t1, voice, 'Canción Original','Tiempo (ms)','', (0,5432))

    plotSound(t2, impulse, 'Impulso','Tiempo (ms)','', (0,700))
    y,zf = lfilter(impulse, 1, voice, zi=zeros(len2-1))
    y = append(y,zf).astype(float32)
    t3 = arange(len(y+len2))*1/fm2*1000
    plotSound(t3, y, 'Canción Convolusionada','Tiempo (ms)','', (0,6175))
    write('T2/convolucion.wav', fm2, y)


def plotDigit(filename, filepath):
    plotAmpVsTime(file, filename,(100,400))

def resampleSoundFreq(filename):
    (fm, s, n) = getFileValues(filename)
    t = arange(n)*(1/fm)*1000
    rf, rt = resample(s, int((n/fm)*2000000), t=t)
    cosModulación = cos(linspace(0, 870000*2*pi,num=len(rf)))
    rf = multiply(rf, cosModulación) #Traslada la señala 870 kHz
    #plotSound(rt, rf, 'Señal modulada ','Tiempo (ms)', '',(100, 800))
    rf= multiply(rf, cosModulación) #Traslada la señal al rango audible de nuevo
    plotSound(rt, rf, 'Señal restaurada','Tiempo (ms)', '',(100, 800))
    rf, rt = resample(rf, int(len(rf)/2000000*fm), t=rt)
    plotSound(rt, rf, 'Señal reconstruida','Tiempo (ms)', '',(100, 800))

def radio():
    (fm, sm, n) = getFileValues('TP_2 Enunciado/AM870,890,910corregido.wav')
    t = arange(n)*(1/fm)*1000
    mods = [cos(linspace(0, 870000*2*pi*n/fm,num=n)), cos(linspace(0, 890000*2*pi*n/fm,num=n)), cos(linspace(0, 910000*2*pi*n/fm,num=n))]
    rfMods = [multiply(sm, mods[0]), multiply(sm, mods[1]), multiply(sm, mods[2])]
    rf0, rt0 = resample(rfMods[0],int(len(rfMods[0])/2000000*10000), t=t)
    rf1, rt1 = resample(rfMods[1],int(len(rfMods[1])/2000000*10000), t=t)
    rf2, rt2 = resample(rfMods[2],int(len(rfMods[2])/2000000*10000), t=t)
    write('T2/radio870.wav', 10000, rf0.astype(float32))
    write('T2/radio890.wav', 10000, rf1.astype(float32))
    write('T2/radio910.wav', 10000, rf2.astype(float32))


#cumbion = 'T2/Parte I/cumbión.wav'
#impulso = 'T2/Parte I/aplauso.wav'
#convolucionarSonidos(cumbion, impulso)
#filename = '7'
#file = 'T1/Muestras/1/' + filename + 'f1.wav'
#plotDigit(filename, file)
#resampleSoundFreq(file)
#radio()


#_____________________________________________________________________________________________________________________________________________
#TAREA 3
#_____________________________________________________________________________________________________________________________________________

#2 ciclo para que los lóbulos no se toquen.
#2 138 Hz debería ser la longitud de onda de los rectangulos.
#3


def stft(signal, window, fm, overlap):
    fftRst = []
    k = 0
    while k < (len(signal) - len(window)):
        fftTmp = signal[k:(k + len(window))]
        fftTmp *= window
        fftTmp = fft.fft(fftTmp, fm)[:int(fm / 2)]
        fftTmp = abs(fftTmp)
        if (len(fftRst) == 0):
            fftRst = fftTmp
        else:
            for n in arange(0, len(fftTmp)):
                fftRst[n] = fftRst[n] + fftTmp[n]
        k += int(len(window) - overlap)
    return fftRst

def graficarDienteDeSierra():
    fm = 20_000
    f0 = 138
    t = linspace(0,1,fm)
    dientes = sawtooth(2*pi*f0*t)
    tv = int((fm/f0))
    param = 2
    figureLabel = ""
    if(param == 0):
        window = boxcar(int(2*tv))
        overlap = 0
        figureLabel = "Diente de Sierra-Window=Boxcar"
    if(param == 1):
        window = get_window("hann", 8*tv)
        overlap = int(len(window)/2)
        figureLabel = "Diente de Sierra- Window=Hann"
    if(param == 2):
        window = get_window("hamming", 8*tv)
        overlap = int(len(window) / 2)
        figureLabel = "Diente de Sierra-Window=Hamming"

    fftRst = stft(dientes, window, fm, overlap)
    fftRst /= np.max(fftRst) #normalizar la señal
    print(f"f_0={fftRst[138]}, 2f_0={fftRst[276]}, 3f_0={fftRst[414]}, 4f_0={fftRst[552]}")
    figure(figureLabel)
    plot(fftRst)
    xlabel("Freq (Hz)")
    ylabel("Magnitude")
    show()
    return 0

def graficar_A_STFT():
    (fm,s,n)=getFileValues('T1/Muestras/1/af1.wav')
    f0 = 210
    t = arange(n)*(1/fm)*1000
    tv = int(fm/f0)
    window = get_window("hamming", 4*tv)
    fftRst = stft(s, window, fm,int(len(window)/4))
    fftRst /= np.max(fftRst) #normalizar la señal
    print(f"f_0={fftRst[f0]}, 2f_0={fftRst[f0*2]}, 3f_0={fftRst[f0*3]}, 4f_0={fftRst[f0*4]}")
    figureLabel = "Masculino"
    figure(figureLabel)
    plot(fftRst)
    xlabel("Freq (Hz)")
    ylabel("Magnitude")
    show()
    return 0

graficarDienteDeSierra()
graficar_A_STFT()



