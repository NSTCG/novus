a=int(input())
b={}
for i in range(a):
    b[str(i)]=input()

for i in range(a):
    t=b[str(i)]
    t=t.split()
    d=min(int(t[0]),int(t[1]))
    r=d/2
    print(4*r*r)


#	k=input()
#	c=k.split("")
#    print(c)
#    c[1]=int(c[1])
#    d=min(c)
#	b.append(d)
