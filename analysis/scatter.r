# MIT License
# 
# Copyright © 2022-23 Ryland Goldman
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#   
#   The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

do_export <- TRUE

# Load packages
library(readr)
library(plotly)
library(plyr)
library(RSelenium)

# Source data from CSV
csv_data <- data.frame(read_csv("/Users/rylandgoldman/Downloads/data.csv"))

# Load data into data frames
m8192np <- data.frame(Time=csv_data$X8192.NumPy, Framework="NumPy",Bodies=8192)
m8192cp <- data.frame(Time=csv_data$X8192.CuPy, Framework="CuPy",Bodies=8192)
m8192cc <- data.frame(Time=csv_data$X8192.OpenCL..CPU., Framework="OpenCL CPU",Bodies=8192)
m8192cg <- data.frame(Time=csv_data$X8192.OpenCL..GPU., Framework="OpenCL GPU",Bodies=8192)
m16384np <- data.frame(Time=csv_data$X16384.NumPy, Framework="NumPy",Bodies=16384)
m16384cp <- data.frame(Time=csv_data$X16384.CuPy, Framework="CuPy",Bodies=16384)
m16384cc <- data.frame(Time=csv_data$X16384.OpenCL..CPU., Framework="OpenCL CPU",Bodies=16384)
m16384cg <- data.frame(Time=csv_data$X16384.OpenCL..GPU., Framework="OpenCL GPU",Bodies=16384)
m32768np <- data.frame(Time=csv_data$X32768.NumPy, Framework="NumPy",Bodies=32768)
m32768cp <- data.frame(Time=csv_data$X32768.CuPy, Framework="CuPy",Bodies=32768)
m32768cc <- data.frame(Time=csv_data$X32768.OpenCL..CPU., Framework="OpenCL CPU",Bodies=32768)
m32768cg <- data.frame(Time=csv_data$X32768.OpenCL..GPU., Framework="OpenCL GPU",Bodies=32768)
m65536np <- data.frame(Time=csv_data$X65536.NumPy, Framework="NumPy",Bodies=65536)
m65536cp <- data.frame(Time=csv_data$X65536.CuPy, Framework="CuPy",Bodies=65536)
m65536cc <- data.frame(Time=csv_data$X65536.OpenCL..CPU., Framework="OpenCL CPU",Bodies=65536)
m65536cg <- data.frame(Time=csv_data$X65536.OpenCL..GPU., Framework="OpenCL GPU",Bodies=65536)
m131072np <- data.frame(Time=csv_data$X131072.NumPy, Framework="NumPy",Bodies=131072)
m131072cp <- data.frame(Time=csv_data$X131072.CuPy, Framework="CuPy",Bodies=131072)
m131072cc <- data.frame(Time=csv_data$X131072.OpenCL..CPU., Framework="OpenCL CPU",Bodies=131072)
m131072cg <- data.frame(Time=csv_data$X131072.OpenCL..GPU., Framework="OpenCL GPU",Bodies=131072)
m262144np <- data.frame(Time=csv_data$X262144.NumPy, Framework="NumPy",Bodies=262144)
m262144cp <- data.frame(Time=csv_data$X262144.CuPy, Framework="CuPy",Bodies=262144)
m262144cc <- data.frame(Time=csv_data$X262144.OpenCL..CPU., Framework="OpenCL CPU",Bodies=262144)
m262144cg <- data.frame(Time=csv_data$X262144.OpenCL..GPU., Framework="OpenCL GPU",Bodies=262144)

# Combine data frames
all_data <- data.frame(Time = c(), Framework = c(), Bodies = c())
all_data <- rbind(all_data, m8192np)
all_data <- rbind(all_data, m8192cp)
all_data <- rbind(all_data, m8192cc)
all_data <- rbind(all_data, m8192cg)
all_data <- rbind(all_data, m16384np)
all_data <- rbind(all_data, m16384cp)
all_data <- rbind(all_data, m16384cc)
all_data <- rbind(all_data, m16384cg)
all_data <- rbind(all_data, m32768np)
all_data <- rbind(all_data, m32768cp)
all_data <- rbind(all_data, m32768cc)
all_data <- rbind(all_data, m32768cg)
all_data <- rbind(all_data, m65536np)
all_data <- rbind(all_data, m65536cp)
all_data <- rbind(all_data, m65536cc)
all_data <- rbind(all_data, m65536cg)
all_data <- rbind(all_data, m131072np)
all_data <- rbind(all_data, m131072cp)
all_data <- rbind(all_data, m131072cc)
all_data <- rbind(all_data, m131072cg)
all_data <- rbind(all_data, m262144np)
all_data <- rbind(all_data, m262144cp)
all_data <- rbind(all_data, m262144cc)
all_data <- rbind(all_data, m262144cg)

# Create mean/2*sem plots
data_mean <- ddply(all_data, c("Framework", "Bodies"), summarise, Time = mean(Time))
data_sem <- ddply(all_data, c("Framework", "Bodies"), summarise, Time = 2*sd(Time)/sqrt(100))
data <- data.frame(data_mean, data_sem$Time)
data <- rename(data, c("data_sem.Time" = "sem"))
data$Bodies <- as.factor(data$Bodies)

# Load figure
fig <- plot_ly(data = data[which(data$Framework == 'NumPy'),], x = ~Bodies, y = ~Time, type = 'scatter', mode='lines+markers',  name='NumPy', error_y = ~list(array = sem, color = '#000000'), yaxis = list(type = "log")) %>% add_trace(data = data[which(data$Framework == 'CuPy'),], name = 'CuPy') %>% add_trace(data = data[which(data$Framework == 'OpenCL CPU'),], name = 'OpenCL CPU') %>% add_trace(data = data[which(data$Framework == 'OpenCL GPU'),], name = 'OpenCL GPU')

# Adjust layout
fig <- layout(fig, yaxis = list(title = list(text="Time (seconds, log scale)",font=list(family = "Scala")), type = "log"), xaxis = list(title = list(text="Number of Bodies (log scale)", font=list(family = "Scala"))), title=list(text="Number of Bodies vs. Compute Time", font=list(family = "Scala")), legend=list(font=list(family = "Scala")), plot_bgcolor  = "rgba(0, 0, 0, 0)",paper_bgcolor = "rgba(0, 0, 0, 0)", margin = list(   l = 50,   r = 50,   b = 100,   t = 100,   pad = 4 ))

# Display figure
fig

# Export figure
if(do_export){
  export(fig, file = " plot.svg", selenium = RSelenium::rsDriver(browser = "firefox"))
}

# After file is exported, run `svgexport *.svg out.png 8x` in terminal to create png
