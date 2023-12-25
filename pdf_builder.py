#!/usr/bin/env python
# coding: utf-8

# In[57]:


beginning_string = """<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<style>
.Font{
font-size: 25px;
text-align: justify;
text-justify: inter-word;
}
.good_border{
  border: 2px solid #aaaaaa;
  border-radius: 5px;
}
.image{
width:80%;
}
.h-line{
width:100%;
height: 0px;
border-bottom: 1px solid #aaaaaa;
margin-top: 20px;
margin-bottom: 20px;
background: #aaaaaa;
}
</style>
</head>
<body>
<div class='container mt-4' id = "btn">
<div class ='row-content'>
<div class = 'col-12 d-flex justify-content-center'>
<button class="btn btn-lg btn-primary" id="submit" type = "button" onclick="printDiv('print')" >Get PDF</button>
</div>
</div>
</div>
<div class="container mt-5 good_border">
<div class="row row-content" id="print">
<div class="col-12">"""

end_string = """</div>
</div>
</div>
</body>
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
<script>
		function printDiv(divName){
			$("#btn").remove();
			//var printContents = document.getElementById(divName).innerHTML;
			//var originalContents = document.body.innerHTML;

			//document.body.innerHTML = printContents;

			window.print();
			
			history.go(0);
			//document.body.innerHTML = originalContents;

		}
	</script>
</html>
"""
hr = ''' <div class="h-line"></div>'''
text_start = """
<p class="Font mx-4">"""
text_end = """</p>
"""
img_start='''
<img src = "'''
img_end = '''" class="ml-5 mr-2 image" /><br/>
'''
table_start = """
<table class="table Font">
  <thead class="thead-light">
"""
table_end ="""</tbody>
</table>
"""


# In[46]:


def get_table_heading(head1, head2):
    return '<tr><th scope="col">#</th><th scope="col">' +head1+ '</th><th scope="col">' +head2+ '</th></tr></thead><tbody>'


# In[34]:


def get_table_row(counter,row1, row2):
    return '<tr><th scope="row">'+str(counter)+'</th><td>'+row1+'</td><td>'+row2+'</td></tr>'


# In[26]:


def get_par(text):
    global text_start
    global text_end
    
    return text_start + text + text_end


# In[27]:


def get_img(path):
    global img_start
    global img_end
    return img_start + path + img_end


# In[63]:


# this file is for html compilation

def convert_to_html(exp_dict, dif_dict, info):
    # 1.print info of the candidate
    # 2.print all expansions
    # 3.print all differenciations
    
    # structure of info:
    # info[0] contains Name of student
    # info[1] roll number
    # info[2] number of questions attempted
    
    global beginning_string
    global end_string
    global text_start
    global text_end
    global table_start
    global table_end
    global hr
    complete_html = beginning_string
    
    # use text_start + text + text_end for texts
    # use text_start + text_end for leaving single_line
    
    single_line = text_start +text_end
    
    complete_html += single_line
    
    for line in info[:-1]:
        complete_html += get_par(line)
        complete_html += single_line
    complete_html += get_par("Number of questions attempted: "+info[-1])
    complete_html += single_line
    complete_html += single_line
    complete_html += hr
    counter = 0
    for question in exp_dict:
        complete_html += get_par(question['question'])
        for line in question['answer']:
            if line != 'IMAGEFLAG':
                complete_html += get_par(line)
            else:
                fpath = "J:\\work_station\\cache\\"+question['question']+"\\"+str(counter)+"_"+info[1]+".jpg"
                complete_html += single_line
                complete_html += get_img(fpath)
                complete_html += single_line
                counter += 1
        complete_html += single_line
        complete_html += hr
    
    
    for question in dif_dict:
        complete_html += get_par(question['question'])
        headings = list(question['answer'].keys())
        complete_html += table_start
        complete_html += get_table_heading(headings[0],headings[1])
        con = 1
        for i in range(len(min(question['answer'][headings[0]],question['answer'][headings[1]]))):
            complete_html += get_table_row(con,question['answer'][headings[0]][i],question['answer'][headings[1]][i])
            con+=1
        complete_html += table_end
        complete_html += hr
    
    complete_html += end_string
    file = open("J:\\work_station\\student_papers\\"+info[1]+"_digital.html","w")
    file.write(complete_html)
    file.close()

    return

