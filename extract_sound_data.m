samples = 16;
[x,fs] = audioread('C:/Users/danis/Desktop/drv_drow/sounds/1.wav');
%info_alarm = audioinfo('alarm.wav');

audiowrite('C:/Users/danis/Desktop/drv_drow/sounds/1_sampled.wav',x,44100,'BitsPerSample',16);
info_sampled = audioinfo('C:/Users/danis/Desktop/drv_drow/sounds/1_sampled.wav');
