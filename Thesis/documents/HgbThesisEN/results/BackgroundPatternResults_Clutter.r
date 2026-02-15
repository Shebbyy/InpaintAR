png(filename = "BackgroundClutter.png", 
    width = 2000,   
    height = 1600, 
    res = 225)

x <- c("FMM", "NLTM", "ELTM", "LaMa")
y <- c(3, 1.27, 2.67, 5.4)

bp <- barplot(y, 
              names.arg = x, 
              horiz = TRUE, 
              col = "darkblue", 
              xlab = "Clutter Reduction Metric (0-100%)", 
              main = "Clutter Reduction in % (Higher = Better)", 
              xlim = c(0, 10),
              cex.axis = 2,
              cex.names = 2,
              cex.lab = 1.9,
              cex.main = 2.5,
              space = 0.25)

text(x = y + 0.5,  
     y = bp,
     labels = y,
     cex = 2)

dev.off()