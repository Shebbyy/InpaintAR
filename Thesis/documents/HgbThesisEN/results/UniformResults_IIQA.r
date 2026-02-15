png(filename = "UniformQuality.png", 
    width = 2000, 
    height = 1600,
    res = 225)

x <- c("FMM", "NLTM", "ELTM", "LaMa")
y <- c(66, 63, 68, 95)

bp <- barplot(y, 
              names.arg = x, 
              horiz = TRUE, 
              col = "darkblue", 
              xlab = "IIQA normalized (0-100)", 
              main = "Image Inpainting Quality Assessment (Higher = Better)", 
              xlim = c(0, 110),
              cex.axis = 2,
              cex.names = 2,
              cex.lab = 1.9,
              cex.main = 1.9,
              space = 0.25)

text(x = y + 3.25,   
     y = bp,    
     labels = y,
     cex = 2)

dev.off()