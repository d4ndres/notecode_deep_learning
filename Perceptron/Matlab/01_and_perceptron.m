clc, clear all, close all;
p = [ 0 1 0 1; 0 0 1 1];
t = [ 0 0 0 1 ];

%% Entrenamiento

w = rand(1,2);
b = rand(1);

for epoch=1:50
    for r=1:length(p)
        a = hardlim(w*p(:,r)+b);
        e(:,r)=t(:,r)-a;
        w=w+e(:,r)*p(:,r)';
        b=b+e(:,r);
    end  
end

%% grafica de entrenamiento
plot(p(1,1),p(2,1),"bo",p(1,2),p(2,2),"bo"), axis([-1 2 -1 2])
hold on
grid on
plot(p(1,3),p(2,3),"bo",p(1,4),p(2,4),"bx")

%% grafica frontera

p1 = -2:0.01:3;
p2 = -(w(1,1)/w(1,2))*p1 -b(1)/w(1,2);
plot(p1,p2)
