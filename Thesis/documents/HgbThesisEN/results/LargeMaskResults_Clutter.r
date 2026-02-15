png(filename = "LaMaClutter.png", 
    width = 2000,   
    height = 1600, 
    res = 225)

x <- c("FMM", "NLTM", "ELTM", "LaMa")
y <- c(4.2, 1.26, 10.33, 5.55)

bp <- barplot(y, 
              names.arg = x, 
              horiz = TRUE, 
              col = "darkblue", 
              xlab = "Clutter Reduction Metric (0-100%)", 
              main = "Clutter Reduction in % (Higher = Better)", 
              xlim = c(0, 15),
              cex.axis = 2,
              cex.names = 2,
              cex.lab = 1.9,
              cex.main = 2.5,
              space = 0.25)

text(x = y + 0.75,  
     y = bp,
     labels = y,
     cex = 2)

dev.off()