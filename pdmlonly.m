%% Phase-Distortion-Synthesis
% Die Phase-Distortion-Synthesis(PD) ist ein Spezialfall der Frequenzmodulation. 
% Hierbei ist die Modulationsfunktion hart an die Frequenz der Carrierfrequenz 
% synchronisiert. In den hier betrachteten Fällen wird dabei die lineare Phase 
% einer reinen Sinusschwingung so verzerrt, dass sich die Form so verändert, dass 
% sie bekannten, obertonreichen Wellenformen wie Sägezahn- oder Rechteckschwingung 
% ähneln.
% 
% Die Synthesemethode wurde duch Casio Mitte der 80er Jahre eingeführt und findet 
% z.B in den Synthezisern der CZ-Reihe (z.B. CZ-101) Anwendung. Während bei den 
% Casiogeräten Wavetable genutzt wurden, um die Verzerrung zu modellieren, verwende 
% ich in meinem Miniprojekt stückweise lineare Funktionen und berechne Anhand 
% dieser die zugehörigen Werte einer Sinusfunktion gleicher Frequenz.
% 
% Im Gegensatz zur Phasen- oder Frequenzmodulation entstehen bei der PD relativ 
% einfache lineare Spektren. Während die Phasen- und Frequenzmodulation nicht 
% lineare Spektren erzeugt, die durch Besselfunktionen beschrieben werden, ähneln 
% die Spektren der PD denen subtraktiver Synthesemethoden.
% 
% Durch geeignete Phasenfunktionen und Modulation ihrer Paramter, lässt sich 
% das Verhalten eines Tiefpassfilter nachbilden.
%% Theoretische Betrachtung der Phasenfunktionen und Hüllkurven

Fs = 1000;     %Hz
Ts = 1/Fs;      %s
To = 1;         %s
t = 0:Ts:To-Ts; %s

fsin = 5;     %Hz
Tsin = 1/fsin;  %s
set(0,'DefaultLineLineWidth',2);
%% 
% Darstellung des Signalverlaufs der Phasenfunktionen, sowie des resuliterenden 
% Signalverlaufs des verzerrten Sinussignals und das zugehörigen Amplitudenspektrum.

xamt =0.95;
yamt = 0.5;
plotsaw(t, fsin, xamt, yamt, Fs, Tsin);
%%
width = 0.9;
skew = 0;
plot_square(t, fsin, width, skew, Fs, Tsin);
%% 
% Demonstration der AD-Hüllkurven

att = [0 .1 .5]; %s
dec = [.3 .1 .1]; %s
plotenvelopes(t, att, dec);
%% 
% Funktionsweise der Hüllkurven für Zeit-Frames

t = 0:Ts:1-Ts;                              %s [0;1]
env = adEnvelope(t,[1 .5 2],[1 .5 2],1);    %V
figure("Position",[1 1 1200 300]);
subplot(1,2,1);
plot(t,env)
title("Zeitverlauf Hüllkurve Zeit Intervall 0s <= t < 1s");
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
t = t+1;                                    %s [1;2]
env = adEnvelope(t,[1 .5 2],[1 .5 2],0);    %V
subplot(1,2,2);
plot(t,env)
title("Zeitverlauf Hüllkurve 1s <= t < 2s");
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
%% 
% 
%% 
% *Demonstration: Phase-Distortion-Syntheziser spielbar via MIDI-Keyboard und 
% Controller*
% 
% Initialisieren der Midi- und Audiogeräte.

Fs = 44100; %Hz
Ts = 1/Fs;  %s
BufferSize = 1024;

midiInput = mididevice(3);
deviceWriter = audioDeviceWriter(Fs);
%% 
% Initale Paramter

frq = 440;                  %Hz
xamt = 0.5;                 %[0;1]
yamt = 0.5;                 %[0;1]
width = 0.0;                %[0;1]
skew = 0.0;                 %[0;1]
tAttMax = 2;                %s
tDecMax = 2;                %s
tAtt = repelem(tAttMax,3);  %s
tDec = repelem(tDecMax,3);  %s 
attn = zeros(2,2);          %[-1;1]
trg = false;
mix = 0.5;                  %[0;1]
detune = 1;                 %[1;2]
%% Audio Loop
% MIDI-Events verarbeiten und die empfangenen Parameter setzen.

t = 0:Ts:BufferSize*Ts-Ts;
adEnvelope(t,[0 0 0],[0 0 0],1);
while (true)
    % poll Midi notes and CCs
    msgs = midireceive(midiInput);
    for i=1:numel(msgs)
        msg = msgs(i);
        if isNoteOn(msg)
            note = msg.Note;
            frq = midiNote2frq(note);
            % tigger envelopes
            adEnvelope(t,tAtt,tDec,1);
        elseif isCC(msg)
            CC = msg.CCNumber;
            switch CC
                % xamt [0;1]
                case 102
                    CCval = msg.CCValue/127;
                    xamt = CCval;
                % yamt [0;1]
                case 103
                    CCval = msg.CCValue/127;
                    yamt = CCval;
                % atn xamt [-1;1]
                case 104
                    CCval = (msg.CCValue/127)*2-1;
                    attn(1,1) = CCval;
                % atn yamt [-1;1]
                case 105
                    CCval = (msg.CCValue/127)*2-1;
                    attn(1,2) = CCval;
                % width [0;1]
                case 106
                    CCval = msg.CCValue/127;
                    width = CCval;
                % skew [0;1]
                case 107
                    CCval = msg.CCValue/127;
                    skew = CCval;
                % atn width [-1;1]
                case 108
                    CCval = (msg.CCValue/127)*2-1;
                     attn(2,1) = CCval;
                % atn skew [-1;1]
                case 109
                    CCval = (msg.CCValue/127)*2-1;
                    attn(2,2) = CCval;
                % env1 att [0;1]
                case 110
                    CCval = msg.CCValue/127;
                    tAtt(1) = tAttMax*CCval;
                % env1 dec [0;1]
                case 111
                    CCval = msg.CCValue/127;
                    tDec(1) = tDecMax*CCval;
                % env2 att [0;1]
                case 112 
                    CCval = msg.CCValue/127;
                    tAtt(2) = tDecMax*CCval;
                % env2 dec [0;1]
                case 113
                    CCval = msg.CCValue/127;
                    tDec(2) = tDecMax*CCval;
                % env3 att [0;1]
                case 114
                    CCval = msg.CCValue/127;
                    tAtt(3) = tDecMax*CCval;
                % env3 dec [0;1]
                case 115
                    CCval = msg.CCValue/127;
                    tDec(3) = tDecMax*CCval;
                % mix between saw and square [0;1]
                case 116
                    CCval = msg.CCValue/127;
                    mix = CCval;
                % detune 2nd Oscialtor (square) [1;2]
                case 117
                    CCval = msg.CCValue/127+1;
                    detune = CCval;
            end %switch     
        end % if
    end % for   
%% 
% Generieren der Hüllkurven und Anwenden auf die Parameter der Phasenfunktionen. 
% 
% Generieren der Phasenfunktionen und der Sinussignale. Anschließendes mixen 
% der beiden Signale (Sägezahn, Rechteck) und senden an das Audiodevice.
% 
% Wenn alle Hüllkurven auf 0 gefallen sind wird der Zeitvektor zurückgesetzt.

    % generate envelopes
    env = adEnvelope(t,tAtt,tDec,0);
    % apply modulation to paramters and clip if not in [0;1]
    xmod = clip(attn(1,1).*env(2)+xamt);
    ymod = clip(attn(1,2).*env(2)+yamt);
    wmod = clip(attn(2,1).*env(3)+width);
    smod = clip(attn(2,2).*env(3)+skew);
    % generate phases
    phase1 = pd2(t,frq,xmod,ymod);
    phase2 = pd3(t,detune*frq,wmod,smod);
    % mix the distorted sine signals
    sig = mix*sin(phase2)+(1-mix)*sin(phase1);
    % apply envelope
    deviceWriter(0.5*(env(1,:).*sig)');
    % reset time if all envelopes are 0
    if any(env(:))
        t = t + BufferSize*Ts;
    else
        t = 0:Ts:BufferSize*Ts-Ts;
    end    
end % while
%% Funktionen
%% Phasenfunktionen
% Phasenfunktion zur generierung sägezähnähnlicher Funkionen.

function phi = pd2(t,f,x_amt,y_amt)
if x_amt < 0 || x_amt > 1
    error("x_amt must be between 0 and 1");
end
if y_amt < 0 || y_amt > 1
    error("y_amt must be between 0 and 1");
end
T=1/f;
tmod = mod(t,T);

x_off = T/2-x_amt*T/2;
y_off = pi-y_amt*pi;

if x_amt == 1
    slope1 = 0;
else
    slope1 = y_off./x_off;
end

if x_amt == 0
    slope2 = 0;
else
    slope2 = (pi-y_off) ./ (T/2-x_off);
end

phi = tmod.*slope1 .* (tmod < x_off) ...
    + ((tmod-x_off).*slope2+y_off) .* (tmod >=x_off & tmod <= T-x_off)...
    + ((tmod-(T-x_off)).*slope1+(2*pi-y_off)) .* (tmod > (T-x_off));
end
%% 
% Phasenfunktion zur generierung rechteckähnlicher Funkionen.

function phi = pd3(t,f,width,skew)
if width < 0 || width > 1
    error("width must be between 0 and 1");
end
if skew < 0 || skew > 1
    error("skew must be between 0 and 1");
end
T=1/f;
tmod = mod(t,T);

x_off = T/4 - width * T/4;
y_off = pi/2 - skew * pi/2;

if width == 0 
    phi = tmod*2*pi/T;
    return
elseif width == 1
    phi = pi/2 .* (tmod < T/2) ...
        + 3/2*pi .* (tmod <+T/2);
    return
end
slope1 = y_off./x_off;
slope2 = ((pi-y_off)-y_off) / ((T/2-x_off)-x_off);
slope3 = (pi -(pi-y_off)) / (T/2 - (T/2-x_off));

phi = tmod*slope1 .* (tmod < x_off) ...
    + ((tmod-x_off).*slope2+y_off) .* (tmod>=x_off & tmod < T/2-x_off) ...
    + ((tmod-(T/2-x_off)).*slope3+pi-y_off) .* (tmod >= T/2-x_off & tmod< T/2+x_off) ...
    + ((tmod-(T/2+x_off)).*slope2+pi+y_off) .* (tmod >= T/2+x_off & tmod < T-x_off)...
    + ((tmod-(T-x_off)).*slope1+2*pi-y_off) .* (tmod >= T-x_off);

end
%% AD-Envelope

function envelope = adEnvelope(t,att,dec,trig)
persistent t0;
persistent tEnd;
persistent isActive;
persistent elapsed;
persistent attslope
persistent decslope
% initialize static variables
if isempty(t0) || isempty(tEnd) || isempty(isActive) || isempty(elapsed) ...
        || isempty(attslope) || isempty(decslope) 
    t0 = 0;
    tEnd = 0;
    isActive = 0;
    elapsed = 0;
    attslope = 0;
    decslope = 0;
end
if numel(att) ~= numel(dec)
    error("dimensions of att and dec must agree");
end
% tigger the envelope, reset parameters
if trig
    t0 = t(1);
    tEnd = t0+att+dec;
    isActive = true;
    elapsed = 0;
    attslope(att == 0) = 0;
    attslope(att ~= 0) = 1./att(att ~= 0);
    decslope(dec == 0) = 0;
    decslope(dec ~=0) = -1./dec(dec ~= 0);
    t = repmat(t,size(att,2),1);
end
% return 0 if not active
if ~isActive
    envelope = zeros(size(att,1),numel(t));
% else return the part of envelope
else
    envelope = attslope'.*(t-t0) .* (t-t0 <= att') ...
             + ((decslope'.*(t-t0-att')+1) .* (t-t0 >att' & t-t0 <=att'+dec')) ...
             + 0 .* (t-t0 > att'+dec');
    elapsed = t(end) - t0;
end
% reset if time is elapsed
if tEnd <= t0 + elapsed
     isActive = false;
end


end
%% Funktionen zum parsen von MIDI-Nachrichten.

function yes=isNoteOn(msg)
yes = msg.Type == midimsgtype.NoteOn && msg.Velocity > 0;
end
function yes=isCC(msg)
yes = msg.Type == midimsgtype.ControlChange;
end
%% Konvertieren von MIDI-Noten in Frequenzen

function frq = midiNote2frq(note)
frq = 440 * 2^((note-69)/12);
end
%% Clipen von Werten < 0 und >= 1

function y = clip(values)
values(values < 0) = 0;
values(values >= 1) = 0.99;
y = values;
end
%% *Funktionen zum Plotten der Ergebnisse*
% Plotten der rechteckähnlichen Funktion

function plot_square(t, fsin, width, skew, Fs, Tsin)
phase1 = pd3(t,fsin,width,skew);
sig1 = sin(phase1);
spec = fft(sig1);
Nf = numel(spec);
df = Fs / Nf;
f = 0:df:Nf/2-df;
spec1 = [spec(1) 2*spec(2:Nf/2)] / Nf;
figure("Position",[1 1 2000 1800]);
% plot phase function
subplot(3,2,1);
plot(t,phase1);
title("Signalverlauf Phasenfunktion");
xlim([0 2*Tsin]);
ylim([0 2*pi]);
yticks([0 pi/2 pi 3/2*pi 2*pi]);
yticklabels({'0' '\pi/2' '\pi' '3/2\pi' '2\pi'});
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
% plot signal
subplot(3,2,3);
plot(t,sig1);
title("Signalverlauf Moduliertes Signal");
xlim([0 2*Tsin]);
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
% plot spectrum
subplot(3,2,5);
stem(f,abs(spec1),"Marker","none","LineWidth",2);
title(["Ausschnitt Amplitudenspektrum des Modulierten Signals" "(10 harmonische)"]);
xlim([0 10*fsin]);
xlabel("Frequenz f [Hz]");
ylabel("|Amplitude| [V]");
% compare to sawtooth MATLAB function
sqr = square(2*pi*fsin*t);
spec = fft(sqr);
Nf = numel(spec);
df = Fs / Nf;
f = 0:df:Nf/2-df;
spec1 = [spec(1) 2*spec(2:Nf/2)] / Nf;
subplot(3,2,4);
plot(t,sqr);
xlim([0 2*Tsin]);
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
title("Zeitverlauf der Referenzschwingung (square)");
subplot(3,2,6);
stem(f,abs(spec1),"Marker","none","LineWidth",2);
title(["Auschnitt des Spektrums der Referenzschwingung" "(10 harmonische)"]);
xlim([0 10*fsin]);
xlabel("Frequenz f [Hz]");
ylabel("|Amplitude| [V]");
end
%% 
% Plotten der sägezahnähnlichen Funktion

function plotsaw(t, fsin, xamt, yamt, Fs, Tsin)
phase1 = pd2(t,fsin,xamt,yamt);
sig1 = sin(phase1);
spec = fft(sig1);
Nf = numel(spec);
df = Fs / Nf;
f = 0:df:Nf/2-df;
spec1 = [spec(1) 2*spec(2:Nf/2)] / Nf;
figure("Position",[1 1 2000 1800]);
% plot phase function
subplot(3,2,1);
plot(t,phase1);
title("Signalverlauf Phasenfunktion");
xlim([0 2*Tsin]);
ylim([0 2*pi]);
yticks([0 pi/2 pi 3/2*pi 2*pi]);
yticklabels({'0' '\pi/2' '\pi' '3/2\pi' '2\pi'});
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
% plot signal
subplot(3,2,3);
plot(t,sig1);
title("Signalverlauf Moduliertes Signal");
xlim([0 2*Tsin]);
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
% plot spectrum
subplot(3,2,5);
stem(f,abs(spec1),"Marker","none","LineWidth",2);
title(["Ausschnitt Amplitudenspektrum des Modulierten Signals" "(10 harmonische)"]);
xlim([0 10*fsin]);
xlabel("Frequenz f [Hz]");
ylabel("|Amplitude| [V]");
% compare to sawtooth MATLAB function
saw = sawtooth(-2*pi*fsin*t);
spec = fft(saw);
Nf = numel(spec);
df = Fs / Nf;
f = 0:df:Nf/2-df;
spec1 = [spec(1) 2*spec(2:Nf/2)] / Nf;
subplot(3,2,4);
plot(t,saw);
xlim([0 2*Tsin]);
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
title("Zeitverlauf der Referenzschwingung (sawtooth)")
subplot(3,2,6);
stem(f,abs(spec1),"Marker","none","LineWidth",2);
title(["Auschnitt des Spektrums der Referenzschwingung" "(10 harmonische)"]);
xlim([0 10*fsin]);
ylim([0 1]);
xlabel("Frequenz f [Hz]");
ylabel("|Amplitude| [V]");
end
%% 
% Plotten der AD-Hüllkurven

function plotenvelopes(t, att, dec)
env=adEnvelope(t,att, dec,1);
figure;
plot(t,env);
title("Zeitverlauf der AD-Hüllkurven");
xlabel("Zeit t [s]");
ylabel("Amplitude [V]");
xlim([0 max(att+dec)]);
end