import os, sys
f1=open(sys.argv[1])
cl=0
d={}
for line in f1:
	d[line.rstrip()]=cl
	cl+=1
f1.close();
dirs=['train','test']
for di in dirs:
	tr_dirs=os.listdir(di)
	for d1 in tr_dirs:
		if os.path.isdir(os.path.join(di,d1)):
			tr_dirs_d=os.listdir(os.path.join(di,d1))
			for dd in tr_dirs_d:
				if os.path.isdir(os.path.join(di,d1,dd)):
					tr_dirs_dd=os.listdir(os.path.join(di,d1,dd))
					for ddd in tr_dirs_dd:
						if ddd.endswith('.phn'):
							f1=open(os.path.join(di,d1,dd,ddd))
							ddd_o=ddd.split('.')[0]+'.lbl'
							f2=open(os.path.join(di,d1,dd,ddd_o),'w')
							out=""
							for line in f1:
								l=line.rstrip().split()
								out+=l[0]+" "+l[1]+" "+str(d[l[2]])+'\n'
							f2.write(out.rstrip())
							f1.close()
							f2.close()
