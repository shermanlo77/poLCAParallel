poLCA.ylik.C <-
function(vp,y) {
    lik <- ylik(vp$vecprobs,
                t(y),
                dim(y)[1],
                length(vp$numChoices),
                vp$numChoices,
                vp$classes
           )
    lik <- matrix(lik,ncol=vp$classes,byrow=TRUE)
    return(lik)
}
