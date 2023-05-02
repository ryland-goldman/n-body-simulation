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

# Function for error propagation calculations
# https://courses.washington.edu/phys431/propagation_errors_UCh.pdf
calc_error <- function(np, sample){
  error_np <- (2*sd(np)/sqrt(100))/mean(np)
  error_sample <- (2*sd(sample)/sqrt(100))/mean(sample)
  sp <- mean(np)/mean(sample)
  sp * (sqrt(error_sample^2 + error_np^2))
}

# Data frame for speedup calculations and error bars
speedup_df <- data.frame(
  Framework=c("CuPy","OpenCL CPU","OpenCL GPU","CuPy","OpenCL CPU","OpenCL GPU","CuPy","OpenCL CPU","OpenCL GPU","CuPy","OpenCL CPU","OpenCL GPU","CuPy","OpenCL CPU","OpenCL GPU","CuPy","OpenCL CPU","OpenCL GPU"),
  Bodies=c(8192,8192,8192,16384,16384,16384,32768,32768,32768,65536,65536,65536,131072,131072,131072,262144,262144,262144),
  Speedup=c(
    mean(m8192cp$Time)/mean(m8192np$Time),mean(m8192cc$Time)/mean(m8192np$Time),mean(m8192cg$Time)/mean(m8192np$Time),
    mean(m8192cp$Time)/mean(m16384np$Time),mean(m16384cc$Time)/mean(m16384np$Time),mean(m16384cg$Time)/mean(m16384np$Time),
    mean(m32768cp$Time)/mean(m32768np$Time),mean(m32768cc$Time)/mean(m32768np$Time),mean(m32768cg$Time)/mean(m32768np$Time),
    mean(m65536cp$Time)/mean(m65536np$Time),mean(m65536cc$Time)/mean(m65536np$Time),mean(m65536cg$Time)/mean(m65536np$Time),
    mean(m131072cp$Time)/mean(m131072np$Time),mean(m131072cc$Time)/mean(m131072np$Time),mean(m131072cg$Time)/mean(m131072np$Time),
    mean(m262144cp$Time)/mean(m262144np$Time),mean(m262144cc$Time)/mean(m262144np$Time),mean(m262144cg$Time)/mean(m262144np$Time)
  ),
  Error = c(
    calc_error(m8192cp$Time,m8192np$Time),calc_error(m8192cc$Time,m8192np$Time),calc_error(m8192cg$Time,m8192np$Time),
    calc_error(m16384cp$Time,m16384np$Time),calc_error(m16384cc$Time,m16384np$Time),calc_error(m16384cg$Time,m16384np$Time),
    calc_error(m32768cp$Time,m32768np$Time),calc_error(m32768cc$Time,m32768np$Time),calc_error(m32768cg$Time,m32768np$Time),
    calc_error(m65536cp$Time,m65536np$Time),calc_error(m65536cc$Time,m65536np$Time),calc_error(m65536cg$Time,m65536np$Time),
    calc_error(m131072cp$Time,m131072np$Time),calc_error(m131072cc$Time,m131072np$Time),calc_error(m131072cg$Time,m131072np$Time),
    calc_error(m262144cp$Time,m262144np$Time),calc_error(m262144cc$Time,m262144np$Time),calc_error(m262144cg$Time,m262144np$Time)
  )
)

# Create plots
data_speedup <- ddply(speedup_df, c("Framework", "Bodies"), summarise, Speedup = (1/Speedup) - 1)
data_error <- ddply(speedup_df, c("Framework", "Bodies"), summarise, Speedup = Error)
data <- data.frame(data_speedup, data_error$Speedup)
data <- rename(data, c("data_error.Speedup" = "Error"))
data$Bodies <- as.factor(data$Bodies)

# Load figure
fig <- plot_ly(data = data[which(data$Framework == 'CuPy'),], x = ~Bodies, y = ~Speedup, type = 'bar', name='CuPy', error_y = ~list(array = Error, color = '#000000'), yaxis = list(type = "log"), base = 1,  marker = list(color = "rgba(255,126,40,1)")) %>% add_trace(data = data[which(data$Framework == 'OpenCL CPU'),], name = 'OpenCL CPU',  marker = list(color = "rgba(53,159,57,1)")) %>% add_trace(data = data[which(data$Framework == 'OpenCL GPU'),], name = 'OpenCL GPU',  marker = list(color = "rgba(213,38,44,1)"))

# Adjust layout
fig <- layout(fig, yaxis = list(title = list(text="Speedup Multiple (log scale)",font=list(family = "Scala")), type = "log"), xaxis = list(title = list(text="Number of Bodies (log scale)", font=list(family = "Scala"))), title=list(text="Number of Bodies vs. Speedup Multiple", font=list(family = "Scala")), legend=list(font=list(family = "Scala")), plot_bgcolor  = "rgba(0, 0, 0, 0)",paper_bgcolor = "rgba(0, 0, 0, 0)", margin = list(   l = 50,   r = 50,   b = 100,   t = 100,   pad = 4 ))

# Display figure
fig

# Export figure
if(do_export){
  export(fig, file = " plot.svg", selenium = RSelenium::rsDriver(browser = "firefox"))
}

# After file is exported, run `svgexport plot.svg out.png 8x` in terminal to generate png
