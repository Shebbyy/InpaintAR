png(filename = "ReflectionPerformance.png", 
    width = 2000, 
    height = 1400,
    res = 225)

x <- c("FMM", "NLTM", "ELTM", "LaMa")

total <- c(11.44, 0.53, 3.49, 0.26)
inpaintOnly <- c(17.21, 0.67, 3.92, 0.31)

y <- rbind(total, inpaintOnly)

par(mar = c(5, 6, 4, 2) + 0.1)

bp <- barplot(y, 
              beside = TRUE, 
              names.arg = x, 
              col = c("darkblue", "steelblue"), 
              ylab = "Frames per Second (FPS)", 
              main = "Framerate (Higher = Better)", 
              ylim = c(0, 20),
              cex.axis = 1.6,
              cex.names = 1.6,
              cex.lab = 1.75,
              cex.main = 2.5,
              space = c(0, 0.3),
              legend.text = FALSE)

text(x = bp,          
     y = y + 1.5,      
     labels = y, 
     cex = 1.6)

legend("topright", 
       legend = c("Total", "Inpainting Only"), 
       fill = c("darkblue", "steelblue"),
       cex = 1.25,      
       pt.cex = 1.25,     
       bty = "o")

dev.off()