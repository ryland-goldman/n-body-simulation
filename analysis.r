library(readr)
library(plotly)
library(plyr)
data1 <- read_csv("/Users/rylandgoldman/Downloads/data.csv")
d1 <- data.frame(data1)

m8192np <- data.frame(Time=d1$X8192.NumPy, Framework="NumPy",Bodies=8192)
m8192cp <- data.frame(Time=d1$X8192.CuPy, Framework="CuPy",Bodies=8192)
m8192cc <- data.frame(Time=d1$X8192.OpenCL..CPU., Framework="OpenCL CPU",Bodies=8192)
m8192cg <- data.frame(Time=d1$X8192.OpenCL..GPU., Framework="OpenCL GPU",Bodies=8192)

m16384np <- data.frame(Time=d1$X16384.NumPy, Framework="NumPy",Bodies=16384)
m16384cp <- data.frame(Time=d1$X16384.CuPy, Framework="CuPy",Bodies=16384)
m16384cc <- data.frame(Time=d1$X16384.OpenCL..CPU., Framework="OpenCL CPU",Bodies=16384)
m16384cg <- data.frame(Time=d1$X16384.OpenCL..GPU., Framework="OpenCL GPU",Bodies=16384)

m32768np <- data.frame(Time=d1$X32768.NumPy, Framework="NumPy",Bodies=32768)
m32768cp <- data.frame(Time=d1$X32768.CuPy, Framework="CuPy",Bodies=32768)
m32768cc <- data.frame(Time=d1$X32768.OpenCL..CPU., Framework="OpenCL CPU",Bodies=32768)
m32768cg <- data.frame(Time=d1$X32768.OpenCL..GPU., Framework="OpenCL GPU",Bodies=32768)

m65536np <- data.frame(Time=d1$X65536.NumPy, Framework="NumPy",Bodies=65536)
m65536cp <- data.frame(Time=d1$X65536.CuPy, Framework="CuPy",Bodies=65536)
m65536cc <- data.frame(Time=d1$X65536.OpenCL..CPU., Framework="OpenCL CPU",Bodies=65536)
m65536cg <- data.frame(Time=d1$X65536.OpenCL..GPU., Framework="OpenCL GPU",Bodies=65536)

m131072np <- data.frame(Time=d1$X131072.NumPy, Framework="NumPy",Bodies=131072)
m131072cp <- data.frame(Time=d1$X131072.CuPy, Framework="CuPy",Bodies=131072)
m131072cc <- data.frame(Time=d1$X131072.OpenCL..CPU., Framework="OpenCL CPU",Bodies=131072)
m131072cg <- data.frame(Time=d1$X131072.OpenCL..GPU., Framework="OpenCL GPU",Bodies=131072)

m262144np <- data.frame(Time=d1$X262144.NumPy, Framework="NumPy",Bodies=262144)
m262144cp <- data.frame(Time=d1$X262144.CuPy, Framework="CuPy",Bodies=262144)
m262144cc <- data.frame(Time=d1$X262144.OpenCL..CPU., Framework="OpenCL CPU",Bodies=262144)
m262144cg <- data.frame(Time=d1$X262144.OpenCL..GPU., Framework="OpenCL GPU",Bodies=262144)

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

data_mean <- ddply(all_data, c("Framework", "Bodies"), summarise, Time = mean(Time))
data_sem <- ddply(all_data, c("Framework", "Bodies"), summarise, Time = 2*sd(Time)/sqrt(100))
data <- data.frame(data_mean, data_sem$Time)
data <- rename(data, c("data_sem.Time" = "sem"))
data$Bodies <- as.factor(data$Bodies)

fig <- plot_ly(data = data[which(data$Framework == 'NumPy'),], x = ~Bodies, y = ~Time, type = 'scatter', mode='lines+markers',  name='NumPy',
               error_y = ~list(array = sem, color = '#000000'), yaxis = list(type = "log"))
fig <- fig %>% add_trace(data = data[which(data$Framework == 'CuPy'),], name = 'CuPy')
fig <- fig %>% add_trace(data = data[which(data$Framework == 'OpenCL CPU'),], name = 'OpenCL CPU')
fig <- fig %>% add_trace(data = data[which(data$Framework == 'OpenCL GPU'),], name = 'OpenCL GPU')
fig <- layout(fig, yaxis = list(title = list(text="Time (seconds, log scale)",font=list(family = "Scala")), type = "log"), xaxis = list(title = list(text="Number of Bodies (log scale)", font=list(family = "Scala"))), title=list(text="Number of Bodies vs. Compute Time", font=list(family = "Scala")), legend=list(font=list(family = "Scala")))
fig
