import pandas as pd
import matplotlib.pyplot as plt
import quandl 
import numpy as np
from sklearn.svm import SVR
import datetime
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import image_final
from tkinter import *
import tkinter.messagebox
from tkinter import ttk 
import tkinter
import cv2
import PIL.Image, PIL.ImageTk

class stock:
    
    
    def __init__(self,root):
        
        def predict_price():
            C=Company.get()
            print(C)
            if C=='TCS':
                CompanyCode=1
            elif C=='LUX':
                CompanyCode=2
            else :
                CompanyCode=3
            entry1=Date.get()
            select=CompanyCode
            entry=entry1
            from sklearn.model_selection import train_test_split
            entry=entry.replace("2018","") 
            entry=entry.replace("-","")
            entry=float(entry)
            quandl.ApiConfig.api_key = 'itoQatoMoCn1z_v2FogK'
            if select==1:
                stock_data=quandl.get('NSE/TCS', start_date='2018-04-01',end_date='2018-04-28')
            elif select==2:
                stock_data=quandl.get('BSE/BOM539542', start_date='2018-04-01',end_date='2018-04-28')
            else:
                stock_data=quandl.get('BSE/BOM539678', start_date='2018-04-01',end_date='2018-04-28')
                        
                        
            dataset=pd.DataFrame(stock_data)
            dataset.to_csv('cool.csv')
            data=pd.read_csv('cool.csv')
            data['Date']=data.Date.str.replace('-','')
            data['Date'] = pd.to_numeric(data.Date.str.replace('2018',''))
            
            x=data.loc[:, 'Date']
            y=data.loc[:,'Open']
        
        
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0,shuffle=False)
            x_train=pd.Series.to_numpy(x_train)
            x_test=pd.Series.to_numpy(x_test)
            
            x_train = np.reshape(x_train,(len(x_train), 1)) 
            x_test = np.reshape(x_test,(len(x_test), 1)) 
            
            svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 'auto')
            
            
            svr_rbf.fit(x_train, y_train)
            svr_rbf.score(x_test,y_test)
            
            linear_mod = linear_model.LinearRegression()
            linear_mod.fit(x_train,y_train)
            linear_mod.score(x_test,y_test)
            plt.scatter(x_train, y_train, color= 'black', label= 'Data')
            	
            plt.plot(x_train, svr_rbf.predict(x_train), color= 'red', label= 'RBF model') 
            plt.plot(x_train,linear_mod.predict(x_train),color='blue',linewidth=3,label='linear model') 
            
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Support Vector Regression')
            plt.legend()
            plt.savefig("stock.png")
            
            svr_reg = svr_rbf.predict(np.array(entry).reshape(-1,1))[0]
            lin_reg = linear_mod.predict(np.array(entry).reshape(-1,1))[0]
            print(svr_reg)
            print(lin_reg)
            
            self.lblPredict1= Label(DataFrame2, font=('Arial',14,'bold'), text="Predicted value by Linear Regression     \t   : ",padx=2,pady=2,bg="Ghost White")
            self.lblPredict1.grid(row=0,column=0,sticky=W)
            
            self.lblPredict2= Label(DataFrame2, font=('Arial',14,'bold'), text=lin_reg,padx=2,pady=2,bg="Ghost White")
            self.lblPredict2.grid(row=0,column=1,sticky=W)
            
            self.lblPredict3= Label(DataFrame2, font=('Arial',14,'bold'), text="Predicted value by Support Vector Regression : ",padx=2,pady=2,bg="Ghost White")
            self.lblPredict3.grid(row=1,column=0,sticky=W)
            
            self.lblPredict4= Label(DataFrame2, font=('Arial',14,'bold'), text=svr_reg,padx=2,pady=2,bg="Ghost White")
            self.lblPredict4.grid(row=1,column=1,sticky=W)
           
            self.btnPredict = Button(DataFrame2, text="Graph", font=('Arial',14,'bold'), height=1, width=16,bd=2,padx=13,command=image_final.show_image)#,command=predict_price(CompanyCode,entry))
            self.btnPredict.grid(row=3,column=2)
        
        
        self.root=root
        self.root.title("prediction")
        self.root.geometry("1350x750+0+0")
        self.root.config(bg="cadet blue")
        Company=StringVar()
        Date=StringVar()
        
        
        MainFrame=Frame(self.root,bg="cadet blue")
        MainFrame.grid()

        DataFrame= Frame(MainFrame,bd=1,width=1350,height=200,padx=20,pady=20,relief=RIDGE,bg="white")
        DataFrame.pack(side=TOP)  
        
        DataFrame2= Frame(MainFrame,bd=1,width=1350,height=600,padx=20,pady=20,relief=RIDGE,bg="cyan")
        DataFrame2.pack(side=BOTTOM)    
        
        
        
        self.lblCompany= Label(DataFrame, font=('Arial',14,'bold'), text="Company",padx=2,pady=2,bg="Ghost White")
        self.lblCompany.grid(row=0,column=0,sticky=W)
        self.cboCompany=ttk.Combobox(DataFrame,font=('Arial',14,'bold'),textvariable=Company,state='readonly',width=26)
        self.cboCompany['value'] = ('', 'TCS', 'LUX', 'Quickheal')
        self.cboCompany.current(1)
        self.cboCompany.grid(row=0,column=1,padx=20,pady=3)
        
        self.lblDate= Label(DataFrame, font=('Arial',14,'bold'), text="Date",padx=2,pady=2,bg="Ghost White")
        self.lblDate.grid(row=0,column=2,sticky=W)
        e= Entry(DataFrame, font=('Arial',14,'bold'), textvariable=Date,bg="ghost white", width = 26)
        
        e.grid(row=0,column=3)
      
        entry1=Date.get()
        self.btnPredict = Button(DataFrame, text="PREDICT", font=('Arial',14,'bold'), height=1, width=16,bd=2,padx=13,command=predict_price)#,command=predict_price(CompanyCode,entry))
        self.btnPredict.grid(row=2,column=2)
        
        
       
        
       

        
        
        
    

        
if __name__=='__main__':
    root=Tk()
    application=stock(root)
    root.mainloop()