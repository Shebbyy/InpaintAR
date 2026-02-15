png(filename = "BackgroundPerformance.png", 
    width = 2000, 
    height = 1400,
    res = 225)

x <- c("FMM", "NLTM", "ELTM", "LaMa")

total <- c(17.04, 0.98, 4.83, 0.4)
inpaintOnly <- c(25.77, 1.1, 5.3, 0.51)

y <- rbind(total, inpaintOnly)

par(mar = c(5, 6, 4, 2) + 0.1)

bp <- barplot(y, 
              beside = TRUE, 
              names.arg = x, 
              col = c("darkblue", "steelblue"), 
              ylab = "Frames per Second (FPS)", 
              main = "Framerate (Higher = Better)", 
              ylim = c(0, 30),
              cex.axis = 1.6,
              cex.names = 1.6,
              cex.lab = 1.75,
              cex.main = 2.5,
              space = c(0, 0.3),
              legend.text = FALSE)

text(x = bp,          
     y = y + 2,      
     labels = y, 
     cex = 1.6)

legend("topright", 
       legend = c("Total", "Inpainting Only"), 
       fill = c("darkblue", "steelblue"),
       cex = 1.25,      
       pt.cex = 1.25,     
       bty = "o")

dev.off()