
#' Process a single line of the data files.
#'
#' @description
#' Produces a list of matricies from a data line.
#' The columns are as follows:
#' time, scanMAC, positionX, positionY, positionZ,
#' orientation, MAC, signalRSSI, channel, router_type
#'
#' @param x A file line
#'
processLine = function(x)
{
  # tokenize line on delimiters ;=,
  tokens = strsplit(x, "[;=,]")[[1]]
  # return null when there are no measurements
  if (length(tokens) == 10) 
    return(NULL)
  # get matrix of measured RSSI
  tmp = matrix(tokens[ - (1:10) ], , 4, byrow = TRUE)
  # add handheld device data and return resulting matrix
  cbind(matrix(tokens[c(2, 4, 6:8, 10)], nrow(tmp), 6, 
               byrow = TRUE), tmp)
}

#' Round the measurement angle to the nearest 45 deg.
#'
#' @param angles a vector of angles; expected 0 - 360
#'
roundOrientation = function(angles) 
  {
  # create a sequence of reference angles
  # 0, 45, 90, ... , 315
  refs = seq(0, by = 45, length  = 9)
  # round angles to the closest reference value
  q = sapply(angles, function(o) which.min(abs(o - refs)))
  c(refs[1:8], 0)[q]
}


#' Load data files for this case study.
#' 
#' @description
#' Produces a data.frame with the following columns
#' "time", "posX", "posY", "orientation", "mac", "signal",
#' "rawTime", "angle"
#' 
#' Only regular access points are kept (router_type==3).
#' Time is converted from milliseconds to seconds.
#' 
#'
#' @param filename The path to the data file
#' @param subMacs A vector of MAC addresses to use in the data
#'
readData = 
  function(filename = './offline.data.txt', 
           subMacs = c("00:0f:a3:39:e1:c0", "00:0f:a3:39:dd:cd", "00:14:bf:b1:97:8a",
                       "00:14:bf:3b:c7:c6", "00:14:bf:b1:97:90", "00:14:bf:b1:97:8d",
                       "00:14:bf:b1:97:81"))
  {
    txt <- readLines(filename)
    lines <- txt[ substr(txt, 1, 1) != "#" ]
    tmp <- lapply(lines, processLine)
    offline <- as.data.frame(do.call("rbind", tmp), 
                             stringsAsFactors= FALSE) 
    
    names(offline) <- c("time", "scanMac", 
                        "posX", "posY", "posZ", "orientation", 
                        "mac", "signal", "channel", "type")
    
    # keep only signals from access points
    offline <- offline[ offline$type == "3", ]
    
    # drop scanMac, posZ, channel, and type - no info in them
    dropVars <- c("scanMac", "posZ", "channel", "type")
    offline <- offline[ , !( names(offline) %in% dropVars ) ]
    
    # drop more unwanted access points
    offline <- offline[ offline$mac %in% subMacs, ]
    
    # convert numeric values
    numVars <- c("time", "posX", "posY", "orientation", "signal")
    offline[ numVars ] <- lapply(offline[ numVars ], as.numeric)
    
    # convert time to POSIX
    offline$rawTime <- offline$time
    offline$time <- offline$time/1000
    class(offline$time) <- c("POSIXt", "POSIXct")
    
    # round orientations to nearest 45
    offline$angle = roundOrientation(offline$orientation)
    
    return(offline)
  }
