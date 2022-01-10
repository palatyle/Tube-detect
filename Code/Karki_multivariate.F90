Program Multivariate 
!--------------------------------------------------------------------------------------------------------------!
! Multivariate - A program to calculate dependent variable temperature based on the random coefficient generated. !
! The combination of coefficient and independent variable that generates the least average residual will be selected. !
!
!.........................
! Explanation of variables:
! Dependent variable: Temperature
! Independent variables: Slope, Aspect, Hillshade, Elevation, NDVI, NDBSI
! Slope : slope of ground determined in ArcMap 10.1 using 10 m NED, range:0 to 90 degree
! Aspect: Direction of maximum slope determined in ArcMap 10.1 using 10 m NED, range:0 to 360 degree !
! Hillshade: Illumination of ground, range: 0 to 255
! Elevation: Altitude of 10 m NED
! NDVI: Normalized Difference Vegetation Index, range: -1 to 1 determined in ENVI 4.8 using Landsat TM 5, band 3 & 4 !
! NDBSI: Normalized Difference Bare Soil Index, range: -1 to 1 determined in ENVI 4.8 using Landsat TM 5, band 4 & 5 !
! Intercept: Multivariate regression Model intercept 
!-------------------------------------------------------------------------------------------------------------! 
implicit none
integer::k,istat
integer::i,n,seedsize,m
integer, dimension(8):: dateVals
integer :: omp_get_num_threads, omp_get_thread_num, nthreads, tid
integer :: t1, t2, rate

integer, dimension(:), allocatable :: seed
real(kind=8),dimension(:),allocatable:: slope,aspect,hillshade,elevation,NDVI,NDBSI,intercept,ET,Res,temp,xx,yy 
real(kind=8)::maxSlope,minSlope,maxAspect1,minAspect1,maxAspect2, minAspect2, maxHillshade,minHillshade,maxElevation,&
minElevation,maxNDVI,minNDVI,maxNDBSI,minNDBSI,maxIntercept,minIntercept 
real(kind=8),dimension(:),allocatable:: coffSlope,coffAspect1,coffAspect2,coffHillshade,coffElevation,&
coffNDVI,coffNDBSI, coffIntercept 
real(kind=8)::ranslope,ranAspect1,ranAspect2,ranHillshade,ranElevation,ranNDVI,ranNDBSI,ranIntercept, minRes
real(kind=8),dimension(:),allocatable:: sumRes, meanRes 
character(len=10)::filename
real::Start, finish,PT

call system_clock(t1, rate)
print*,'Enter the name of Input text file, Eg; filename.txt' 
Read*,filename 
open(unit=99,file=filename,status='old',action='read') 
open(unit=23, file="testOut.txt")
write (23,*) 'Input file=',filename
k=0
istat=0
do while(istat.eq.0)
    read(99,*,iostat=istat)
    k=k+1 
end do
write(*,*) 'Number of Rows=',k 
write (23,*) 'Number of Rows=',k 
rewind(99)

allocate(xx(k),yy(k),slope(k),aspect(k),hillshade(k),elevation(k),NDVI(k),NDBSI(k),intercept(k),ET(k),temp(k),Res(k))
do i=1,k
    read(99,*,end=50) xx(i),yy(i),temp(i),NDVI(i),NDBSI(i),elevation(i),slope(i),aspect(i),hillshade(i) 
end do
50 continue
close(99)
call DATE_AND_TIME(VALUES=dateVals) 
call RANDOM_SEED(SIZE=seedSize) 
allocate(seed(seedSize))
call RANDOM_SEED(GET=seed)
call RANDOM_SEED(PUT=dateVals((9-seedSize):8))

maxSlope=0.1219 
minSlope=-0.07244

maxAspect1=0.01977 
minAspect1=-0.04258

maxAspect2=0.000088 
minAspect2=-0.000047

maxHillshade=0.0845 
minHillshade=-0.01963

maxElevation=0.002773 
minElevation=-0.03677

maxNDVI=9.084 
minNDVI=-10.33

maxNDBSI=61.66 
minNDBSI=25.8

maxIntercept=394.50159
minIntercept=302.889

print*,'How many combinations should I run for a solution?'
read*,m
write (23,*) 'Number of Run=',m 
allocate(sumRes(m),MeanRes(m),coffSlope(m),coffAspect1(m),coffAspect2(m),&
coffHillshade(m),coffElevation(m),coffNDVI(m),coffNDBSI(m),coffIntercept(m))

sumRes(:)=0. 
MeanRes(:)=0.


!$OMP parallel private(tid) 
tid = omp_get_thread_num()
!$OMP do private(i, n, ET, ranSlope, ranAspect1, ranAspect2, ranHillshade,ranElevation,ranNDVI,ranNDBSI,ranIntercept) !, coffSlope, coffAspect, !!$OMP coffHillshade, coffElevation, coffNDVI, coffNDBSI, coffIntercept)
do i=1,m

    call random_number(ranSlope)

    coffSlope(i)=ranSlope*(maxSlope-minSlope)+minSlope

    call random_number(ranAspect1) 
    coffAspect1(i)=ranAspect1*(maxAspect1-minAspect1)+minAspect1

    call random_number(ranAspect2) 
    coffAspect2(i)=ranAspect2*(maxAspect2-minAspect2)+minAspect2

    call random_number(ranHillshade) 
    coffHillshade(i)=ranHillshade*(maxHillshade-minHillshade)+minHillshade

    call random_number(ranElevation) 
    coffElevation(i)=ranElevation*(maxElevation-minElevation)+minElevation

    call random_number(ranNDVI) 
    coffNDVI(i)=ranNDVI*(maxNDVI-minNDVI)+minNDVI

    call random_number(ranNDBSI) 
    coffNDBSI(i)=ranNDBSI*(maxNDBSI-minNDBSI)+minNDBSI

    call random_number(ranIntercept) 
    coffIntercept(i)=ranIntercept*(maxIntercept-minIntercept)+minIntercept
    do n=1,k
        ET(n)=(coffslope(i)*(slope(n)))+(coffAspect1(i)*(aspect(n)))&
        +(coffAspect2(i)*(aspect(n))*(aspect(n)))+(coffHillshade(i)*(hillshade(n)))&
        +(coffElevation(i)*(elevation(n)))+(coffNDBSI(i)*(NDBSI(n)))&
        +(coffNDVI(i)*(NDVI(n)))+coffIntercept(i)
        Res(n)=abs(ET(n)-Temp(n))
        if(Res(n).le.0.)print*, "Not right."
        
        sumRes(i)= sumRes(i) + Res(n)
    end do
    MeanRes(i) = sumRes(i)/k 
    if(sumRes(i).le.0.)print *, "Not right.", sumRes(i) 
end do
!$OMP end do
!$OMP end parallel
call system_clock(t2, rate) 
PT = (t2-t1)/real(rate)
print*,"The lowest average residual is:", minval(MeanRes), "@" ,minloc(MeanRes) 
write (23,*) "The lowest average residual is:",minval(MeanRes), "@",minloc(MeanRes)


print*,"The Multivariate Equation is:","[",coffSlope(minloc(MeanRes)),"*Slope]+[",coffAspect1(minloc(MeanRes)),"*Aspect]+[",&
coffAspect2(minloc(MeanRes)),"*Aspect*Aspect]+[",coffHillshade(minloc(MeanRes)),"*Hillshade]+[",&
coffElevation(minloc(MeanRes)),"*Elevation]+[",& 
coffNDVI(minloc(MeanRes)),"*NDVI]+[",coffNDBSI(minloc(MeanRes)),"*NDBSI]+[",coffIntercept(minloc(MeanRes)),"]"
print*,'Parallel Computation Time :',PT,'seconds'

write (23,*)"The Multivariate Equation :","(",coffSlope(minloc(MeanRes)),"*[Slope])+(",coffAspect1(minloc(MeanRes)),&
"*[Aspect])+( ",coffAspect2(minloc(MeanRes)),"*[Aspect]*[Aspect])+(",coffHillshade(minloc(MeanRes)),"*[Hillshade])+(",&
coffElevation(minloc(MeanRes)),"*[Elevation])+(", coffNDVI(minloc(MeanRes)),"*[NDVI])+(",coffNDBSI(minloc(MeanRes)),&
"*[NDBSI])+(",coffIntercept(minloc(MeanRes)),")"
print*,'Parallel Computation Time :',PT,'seconds'
write (23,*)'Parallel Computation Time :',PT,'seconds'
close(23)
end program Multivariate