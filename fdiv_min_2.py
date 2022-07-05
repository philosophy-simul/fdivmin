#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:19:51 2022

@author: boruttrpin
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import itertools
import csv
import pandas as pd
import streamlit as st

st.set_page_config(
page_title="f-div min explorer",
page_icon="🔦")


def combine(nr_variables, nr_answers):
    if nr_variables==0 or nr_answers==0:
        return([])
    options=list(itertools.product(range(nr_answers), repeat=nr_variables))
    return(options)

def probs(distr,which=[0,1,2,3,4],conjunction=1):
    #conjunction = 1: search for conjunctions, = 0: search for disjunctions.
    n=int(np.log2(len(distr)))
    truthtable=combine(n,2)
    relevant_rows=[]
    for i,j in enumerate(truthtable):
        valid=1
        valids=[]
        for m in which:
            if type(m)==int:
                if j[m]==1:
                    valid=0
                    valids.append(0)
                else:
                    valids.append(1)
            else: # for negations, go with the float
                m=int(m)
                if j[m]==0:
                    valid=0
                    valids.append(0)
                else:
                    valids.append(1)
        if valid==1 and conjunction==1:
            relevant_rows.append(i)
        if 1 in valids and conjunction==0:
            relevant_rows.append(i)
    return relevant_rows

def sumprobs(distr,whichrows=[0,1]):
    s=0
    for row in whichrows:
        s+=distr[row]
    return s

def condprob(distr,which=([0],1),given=([1],1)):
    #which=([props],conjunction)
    #given=([props],conjunction)
    #calculates: pr(which | given)

    rowsWhich = probs(distr,which[0],which[1])
    rowsGiven = probs(distr,given[0],given[1])
    a=sumprobs(distr,list(set(rowsWhich)&set(rowsGiven)))
    b=sumprobs(distr,rowsGiven)
    if b==0:
        return 0
    else: return a/b
    # return a/b

def relevant_ind_tests(n):
    s=set(range(n))
    ss=[]
    for i in range(1,len(s)):
        ss.append(list(map(set, itertools.combinations(s, i))) )
    ss2=[]
    for i in ss:
        for j in i:
            ss2.append(j)
    to_test = []
    singletons = [i for i in ss2 if len(i)==1]
    for subset_1 in singletons:
        for subset_2 in ss2:
            if subset_1.intersection(subset_2)==set():
                to_test.append([list(subset_1),list(subset_2)])
    return to_test



def numerical_indep_tests(distr,floaterrorlim=1e-4):
    nprop=int(np.log2(len(distr)))
    rel_tests=relevant_ind_tests(nprop)
    pos=[0]
    rel1=[]
    rel2=[]
    for test in rel_tests:
        if test[0]==pos:
            rel1.append(test)
        else:
            rel2.append(rel1)
            rel1=[]
            rel1.append(test)
            pos=test[0]
    rel2.append(rel1)
    condinds=[]
    uncondinds=[]
    condinds_num=[]
    uncondinds_num=[]
    for prop_options in rel2:
        prop=prop_options[0][0]
        for option in prop_options:
            prop1=sumprobs(distr,probs(distr,option[0],1))
            prop2=sumprobs(distr,probs(distr,option[1],1))
            both=list((set(option[0]) | set(option[1])))
            prop3=sumprobs(distr,probs(distr,both,1))
            test1=abs(prop1*prop2-prop3)<floaterrorlim
            uncondinds_num.append(abs(prop1*prop2-prop3))
            if test1==True:
                if not ([option[1],option[0]] in uncondinds):
                    uncondinds.append([option[0],option[1]])
            given=option[1]
            for anotheroption in prop_options:
                anothergiven=anotheroption[1]
                if set(given).issubset(set(anothergiven)) and given!=anothergiven:
                    test=abs(condprob(distr,(prop,1),(given,1))-condprob(distr,(prop,1),(anothergiven,1)))<floaterrorlim

                    condinds_num.append(abs(condprob(distr,(prop,1),(given,1))-condprob(distr,(prop,1),(anothergiven,1))))
                    # print(prop,given,anothergiven)
                    # print(abs(condprob(distr,(prop,1),(given,1))-condprob(distr,(prop,1),(anothergiven,1))))
                    if test==True:
                        if not([list(set(anothergiven)-set(given)),prop,given] in condinds):
                            condinds.append([prop,list(set(anothergiven)-set(given)),given])
                            # print(prop,given,anothergiven)
                            # print(abs(condprob(distr,(prop,1),(given,1))-condprob(distr,(prop,1),(anothergiven,1))))
    return uncondinds,condinds,
# uncondinds_num,condinds_num
# uncondinds: [x] is independent of [y]
# condinds: [x] is conditionally indepenedent of [y] given [z]

def random_distrib(n_prop=3):
    n=2**n_prop
    a=[]
    for i in range(n):
        a.append(np.random.uniform(0,1))
    a=[i/sum(a) for i in a]
    return a

def distr_2(a,p,q):
    return [
        a*p,
        a*(1-p),
        (1-a)*q,
        (1-a)*(1-q)
       ]



def f_of_x(x_state,divergence="kl"):
    if divergence == "kl":
        return x_state * np.log(x_state)
    if divergence == "hel":
        return (1-((x_state)**0.5))**2
    if divergence=="ikl":
        return -np.log(x_state)
    if divergence=="chisq":
        return (x_state - 1)**2




def sq_distance(distr1,distr2):
    suma=0
    for a,b in zip(distr1,distr2):
        suma+=(a-b)**2
    return suma


@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')




def update_by_minimization(prior="rand",uncond_constraints=[.5],cond_constraints=[.5,.5],plotting=1,decims=5):

# uncond_constraints = [
#     ([props],prob,conjunct),
#     ([props],prob,conjunct),
#     ([props],prob,conjunct),
#     ...
#     ]

# cond_constraints = [
#     ([[ante_props], conjunct], [[conseq_props], conjunct], prob),
#     ([[ante_props], conjunct], [[conseq_props], conjunct], prob),
#     ([[ante_props], conjunct], [[conseq_props], conjunct], prob),
#     ...
#     ]
    
    if prior=="rand":
        prior = random_distrib(2)


    divergence="kl"
    divergence2="hel"
    divergence3="ikl"
    divergence4="chisq"
    obj_fun1= lambda x: sum(prior[i]*f_of_x((x[i]/prior[i]),divergence) for i in range(len(prior)))
    obj_fun2= lambda x: sum(prior[i]*f_of_x((x[i]/prior[i]),divergence2) for i in range(len(prior)))
    obj_fun3= lambda x: sum(prior[i]*f_of_x((x[i]/prior[i]),divergence3) for i in range(len(prior)))
    obj_fun4= lambda x: sum(prior[i]*f_of_x((x[i]/prior[i]),divergence4) for i in range(len(prior)))

    nprop=int(np.log2(len(prior)))
    bnds = [(0+1e-320, 1) for i in range(2**nprop)] # open lower bound

    cons = [{'type': 'eq', 'fun': lambda x:  sum(x[i] for i in range(len(prior))) - 1}]
    # every distr has to sum up to 1
    
    if uncond_constraints==[-1]:
        uncond_constraints=[]
    if uncond_constraints==[]:
        rich=prior[0]+prior[1]
    else:
        rich=uncond_constraints[0]
        cons.append({'type': 'eq', 'fun': lambda x:  sum(x[i] for i in [0,1]) - rich})
    if cond_constraints==[]:
        oldGrich=prior[0]/(prior[0]+prior[1])
        oldGpoor=prior[2]/(prior[2]+prior[3])
    else:
        if cond_constraints[0]==-1:
            oldGrich=prior[0]/(prior[0]+prior[1])
        else:
            oldGrich=cond_constraints[0]
            cons.append({'type': 'eq', 'fun': lambda x: sum(x[i] for i in [0]) - oldGrich*( sum(x[j] for j in [0,1]))  })
        if cond_constraints[1]==-1:
            oldGpoor=prior[2]/(prior[2]+prior[3])
        else:
            oldGpoor=cond_constraints[1]
            cons.append({'type': 'eq', 'fun': lambda x: sum(x[i] for i in [2]) - oldGpoor*( sum(x[j] for j in [2,3]))  })

            

    independences_prior=numerical_indep_tests(prior)

    res1  = minimize(obj_fun1, x0 =[0.5]*(2**nprop),bounds=bnds,constraints=cons)
    res2  = minimize(obj_fun2, x0 =[0.5]*(2**nprop),bounds=bnds,constraints=cons)
    res3  = minimize(obj_fun3, x0 =[0.5]*(2**nprop),bounds=bnds,constraints=cons)
    res4  = minimize(obj_fun4, x0 =[0.5]*(2**nprop),bounds=bnds,constraints=cons)

    res1a  = minimize(obj_fun1, x0 =[0.5]*(2**nprop),bounds=bnds,constraints=cons[1:])
    res2a  = minimize(obj_fun2, x0 =[0.5]*(2**nprop),bounds=bnds,constraints=cons[1:])
    res3a  = minimize(obj_fun3, x0 =[0.5]*(2**nprop),bounds=bnds,constraints=cons[1:])
    res4a  = minimize(obj_fun4, x0 =[0.5]*(2**nprop),bounds=bnds,constraints=cons[1:])

    # we minimize functions according to constraints and according to various f divergences.

    independences_post1=numerical_indep_tests(res1.x)
    independences_post2=numerical_indep_tests(res2.x)
    independences_post3=numerical_indep_tests(res3.x)
    independences_post4=numerical_indep_tests(res4.x)
    ys = [prior,res1.x,res2.x,res3.x, res4.x]
    ysa = [prior, res1a.x, res2a.x, res3a.x, res4a.x]

    indeps=[independences_prior,independences_post1,independences_post2,independences_post3,independences_post4]
    successes=[res1.success,res2.success,res3.success, res4.success]
    probprops2=[]
    priorStar=distr_2(rich,oldGrich,oldGpoor)
    for distr in ys:
        probprops=[]
        for i in range(nprop):
            probprops.append(sumprobs(distr,probs(distr,[i],1)))
        probprops2.append(probprops)
    if plotting==1:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,figsize=(10,5))
        # fig=plt.figure()
        x = np.arange(1, 1+2**nprop)

        labels=["Pr(R&O)","Pr(R&Y)","Pr(P&O)","Pr(P&Y)"]
        # Add a table at the bottom of the axes
        rows=("Pr(R)","Pr(O|R)","Pr(O|P)","Pr(O)","Pr(R|O)","Pr(R&O)","Pr(R&Y)","Pr(P&O)","Pr(P&Y)","CS distance","MT distance")
        columnZ=["prior","dkl","hel","ikl","chisq","dkl*","hel*","ikl*","chisq*","prior*"]
        locA="top left"
        allmarginals=[]

        ys=ys+ysa[1:]+[priorStar]
        for distrN in range(len(ys)):
            marginals=[]
            for propN in range(nprop):
                marginals.append(sumprobs(ys[distrN],probs(ys[distrN],[propN])))

            marginals.insert(1,ys[distrN][0]/(ys[distrN][0]+ys[distrN][1]))
            marginals.insert(2,ys[distrN][2]/(ys[distrN][2]+ys[distrN][3]))
            marginals.append(ys[distrN][0]/(ys[distrN][0]+ys[distrN][2]))
            marginals.append(ys[distrN][0])
            marginals.append(ys[distrN][1])
            marginals.append(ys[distrN][2])
            marginals.append(ys[distrN][3])
            if distrN != len(ys)-1:
                marginals.append(sq_distance(ys[distrN],priorStar))
            else:
                marginals.append(-12345)
            if distrN > 0 and distrN < 5:
                marginals.append(sq_distance(ys[distrN],ys[distrN+4]))
            else:
                marginals.append(-12345)
            
            allmarginals.append([str(round(i,decims)) if i>-12345 else "N/A" for i in marginals])
            
            allmarginals1=np.array(allmarginals).T.tolist()
        cell_text=allmarginals1
        # the_table = plt.table(cellText=cell_text,
        #                       rowLabels=rows,
        #                       colLabels=columnZ,
        #                       loc=locA)
        # the_table.scale(2, 2)                
        # plt.subplots_adjust(bottom=.1)


        # Adjust layout to make room for the table:

        plt.xticks(x, labels)
        # plt.title("standard")
        ax1.plot(x, ys[1],label="kl",linestyle="None",marker="v")
        ax1.plot(x, ys[2],label="hel",linestyle="None",marker="^")
        ax1.plot(x, ys[3],label="ikl",linestyle="None",marker="<")
        ax1.plot(x,ys[4],label="chisq",linestyle="None",marker=">")
        ax1.plot(x,prior,label="prior",color="m",marker="o",linestyle="None")
        ax1.plot(x,priorStar,label="prior*",color="k",marker="+",linestyle="None")
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(-.45,1), loc='upper left', borderaxespad=0)
        ax1.text(.75,-.35,"plot 1: probs sum to 1 (constraint)")
        ax2.text(.75,-.35,"plot 2: probs need not sum to 1 (*)")
        ax2.plot(x, ysa[1],label="kl",linestyle="None",marker="v")
        ax2.plot(x, ysa[2],label="hel",linestyle="None",marker="^")
        ax2.plot(x, ysa[3],label="ikl",linestyle="None",marker="<")
        ax2.plot(x,ysa[4],label="chisq",linestyle="None",marker=">")
        ax2.plot(x,prior,label="prior",color="m",marker="o",linestyle="None")
        ax2.plot(x,priorStar,label="prior*",color="k",marker="+",linestyle="None")
        ax2.grid(True)
        updtext=""
        updtext2=""
        updtext3=""
        if uncond_constraints!=[]:
            updtext+="from Pr(R)="+str(round(prior[0]+prior[1],decims))+" to Q(R)= "+str(round(rich,decims))
            if cond_constraints!=[]:
                updtext+=","
            ax2.text(.9, 1.7, 'Updated '+updtext)

        if cond_constraints!=[]:
            if cond_constraints[0]!=-1:
                updtext2+="from Pr(O|R)="+str(round(prior[0]/(prior[0]+prior[1]),decims))+" to Q(O|R)= "+str(round(oldGrich,decims))
                if cond_constraints[1]!=-1:
                    updtext2+=","
                ax2.text(.9, 1.5, 'Updated '+updtext2)

            if cond_constraints[1]!=-1:
                updtext3+="from Pr(O|P)="+str(round(prior[0]/(prior[0]+prior[1]),decims))+" to Q(O|P)= "+str(round(oldGpoor,decims))
                ax2.text(.9, 1.3, 'Updated '+updtext3)
        df = pd.DataFrame(cell_text,columns=columnZ,index=rows)
        st.pyplot(fig)

        st.table(df)
        st.write("prior* is the prior adjusted for the learned constraint.")
        st.write("dkl*, hel*, ikl*, chisq* are minimizations without sum(pr)=1 constraint.")
        st.write("CS distance is squared distance from prior*.")
        st.write("MT distance is squared distance from the corresponding distribution we would get without the constraint that sum(pr)=1.")
        csvf = convert_df(df)
        
        st.download_button(
           "Press to Download the Table",
           csvf,
           "minimization.csv",
           "text/csv",
           key='download-csv'
        )

        plt.subplots_adjust(top=0.555,bottom=0.2,left=0.356,right=0.97,hspace=0.75,wspace=0.11)
        updtex2="_to_"
        if uncond_constraints!=[]:
            updtex2+=str(round(rich,decims))
        else:
            updtex2+="x"
        if cond_constraints!=[]:
            if cond_constraints[0]!=-1:
                updtex2+="_"+str(round(oldGrich,decims))
            else:
                updtex2+="_x"
            if cond_constraints[1]!=-1:
                updtex2+="_"+str(round(oldGpoor,decims))
            else:
                updtex2+="_x"
        else:
            updtex2+="_x_x"
        filename="r_oGr_oGp_"+str(round(prior[0]+prior[1],decims))+"_"+str(round(prior[0]/(prior[0]+prior[1]),decims))+"_"+str(round(prior[2]/(prior[2]+prior[3]),decims))+updtex2
        plt.savefig(filename+".pdf",format="pdf")
        # plt.close()
        tabletext=[[""]+columnZ]
        for i,j in enumerate(allmarginals1):
            tabletext.append([rows[i]]+j)
        with open (filename+".csv","w",newline = "") as csvfile:
            my_writer = csv.writer(csvfile, delimiter = ",")
            my_writer.writerows(tabletext)
        
    return ys,probprops2,indeps

st.write(""" This is a tool to update 2 variable-networks by minimizing some central f-divergences; written by Borut Trpin""")
st.write(""" We assume a network defined by Pr(Rich), Pr(Old|Rich), Pr(Old|Poor)... """)
         
number_a1 = st.slider("Choose Pr(Rich):",0.0,1.0,key="1")
number_p1 = st.slider("Choose Pr(Old|Rich):",0.0,1.0,key="2")
number_q1 = st.slider("Choose Pr(Old|Poor):",0.0,1.0,key="3")

number_a2 = st.radio("Set posterior Q(Rich)?",["no", "yes"],key="4")
if number_a2=="yes":
    number_a2 = st.slider("Choose value:",0.0,1.0,key="5")
else:
    number_a2=-1
    
number_p2 = st.radio("Set posterior Q(Old|Rich)?",["no", "yes"],key="6")
if number_p2=="yes":
    number_p2 = st.slider("Choose value:",0.0,1.0,key="7")
else:
    number_p2=-1

number_q2 = st.radio("Set posterior Q(Old|Rich))?",["no", "yes"],key="8")
if number_q2=="yes":
    number_q2 = st.slider("Choose value:",0.0,1.0,key="10")
else:
    number_q2=-1
    
if st.button('Perform an update by minimizing f-divergence'):
    st.write('To repeat, click again')
    st.write("You can change any parameters above.")
    update_by_minimization(distr_2(number_a1,number_p1,number_q1),[number_a2],[number_p2,number_q2],1)
# with open('corinaExample.csv', 'r') as read_obj:

#     # Return a reader object which will
#     # iterate over lines in the given csvfile
#     csv_reader = csv.reader(read_obj)

#     # convert string to list
#     list_of_csv = list(csv_reader)

# aS=[float(i) for i in list_of_csv[2][1:]]
# pS=[float(i) for i in list_of_csv[3][1:]]
# qS=[float(i) for i in list_of_csv[4][1:]]

# triples=[[a,p,q,a*p+(1-a)*q,a*p/(a*p+(1-a)*q)] for a,p,q in zip(aS,pS,qS) ]
# distrs=[distr_2(triples[i][0],triples[i][1],triples[i][2]) for i in range(len(triples))]
# combs=[i+j for i,j in zip(triples,distrs)]
# priorParams=[i[:3] for i in triples[::4]]

# priors=[]
# j=0
# for i in triples:
#     if j%4==0:
#         a=i[0]
#         p=i[1]
#         q=i[2]
#         prior=distr_2(a,p,q)
#         priors.append(prior)
#     else:
#         if i[0]!=a:
#             unc=[i[0]]
#         else:
#             unc=[]
#         if i[1]!=p:
#             con=[i[1]]
#         else:
#             con=[-1]
#         if i[2]!=q:
#             con.append(i[2])
#         else:
#             con.append(-1)
#         update_by_minimization(prior,unc,con,1)
#     j+=1

    
