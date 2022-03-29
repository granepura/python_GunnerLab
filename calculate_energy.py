#!/usr/bin/env python
# coding: utf-8

# In[ ]:


mon1 = ['B'] #, 'b', 'c', 'd', 'f', 'k', 'm', 'l', 'i', 'w', 'j']
#mon2 = ['A', 'B', 'C', 'D', 'F', 'K', 'M', 'L', 'I', 'V', 'J']
mon3 = ['A'] #, 'e', 'o', 'r', 's', 'n', 'q', 'v', 'h', 'p', 't']
#mon4 = ['G', 'E', 'O', 'R', 'S', 'N', 'Q', 'W', 'H', 'P', 'T']
f = open('fort.38')
d={}
for i in f:
  t = i.split()
  d[t[0]]=float(t[1])
f.close()

f = open('fort.38')
es=[]
vdw=[]
ignore=True
n=0
res=[]
for i in f:
  if i[5] in mon1:
    n+=1
    t = i.split()
    occ = float(t[1])
    if occ>0:
      e = open('energies/'+t[0]+'.opp')
      for j in e:
         if j[11] in mon3:
           k = j.split()
           if float(k[3])<0:
             es.append(round(occ*float(k[2])*d[k[1]],2))
             vdw.append(occ*float(k[3])*d[k[1]])
           else:
             es.append(round(occ*float(k[2])*d[k[1]],2))
             vdw.append(occ*float(k[3])*d[k[1]])
             #vdw.append(0.)
             #if float(k[3])>1 and d[k[1]]>0.9:print(j)
           res.append((t[0], k[1]))
      e.close()
svdw=sorted(vdw)
ses=sorted(es)
Z = [x for _,x in sorted(zip(vdw,res))]
E = [x for _,x in sorted(zip(es,res))]
print(n, sum(es), sum(vdw))
print('vdw',res[vdw.index(min(vdw))], min(vdw))
print('es',res[es.index(min(es))], min(es))
#for i,j in zip(svdw,Z):
#  print(j,i)
sum_vdw=0.
sum_els=0.
A=[]
B=[]
fb = open('B.txt')
fa = open('A.txt')
for i in fb: B.append(int(i))
for i in fa: A.append(int(i))
print(E[0][0][7:10], B[0])
x=-1
for i in range(10):
#  if (int(E[i][0][7:10]) in B) and (int(E[i][1][7:10]) in A):
   print(Z[i], svdw[i], E[i], ses[i])
   print(Z[x], svdw[x], E[x], ses[x])
   sum_vdw+=svdw[i]
   sum_vdw+=svdw[x]
   sum_els+=ses[i]
   sum_els+=ses[x]
   x=x-1
print(sum_vdw, sum_els)

