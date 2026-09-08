! Geometry-only D3 control using the installed QE 7.5 library, not pw.x/SCF.
! Input: the prepared pair_qe_d3.in on stdin. Run all ranks on the same file.
! Uses the same API and MPI startup as QE's dft-d3/test_code.f90.
! Build/run still requires the provider's compatible GNU Fortran module ABI.
program w_qe_d3_component
  use dftd3_api
  use mp_global, only: mp_startup, mp_global_end
  use io_global, only: ionode, ionode_id
  use mp_images, only: intra_image_comm
  use mp, only: mp_bcast
  use constants, only: BOHR_RADIUS_SI, AUTOEV
  implicit none
  integer, parameter :: wp=kind(1.0d0)
  integer :: n, i, j, k, outunit, status, env_status
  character(len=8) :: lifecycle_only
  integer, allocatable :: z(:)
  real(wp), allocatable :: x(:,:,:), g(:,:), u(:,:), displaced(:,:)
  real(wp) :: cell(3,3), stress(3,3), e, ep, em, h, projection
  real(wp) :: abohr
  type(dftd3_input) :: inp
  type(dftd3_calc) :: calc
  ! Full initialization matches mp_global_end; images_only leaves pool/band
  ! communicators uninitialized, causing a post-output MPI_Comm_free failure.
  call mp_startup()
  call get_environment_variable('W_D3_LIFECYCLE_ONLY',lifecycle_only,status=env_status)
  if(env_status == 0 .and. trim(lifecycle_only) == '1') then
    call mp_global_end()
    if(ionode) write(*,'(A)') 'LIFECYCLE_COMPLETE_NO_D3_CALLS'
    stop
  end if
  abohr=BOHR_RADIUS_SI*1.0e10_wp
  if (ionode) read(*,*) n
  call mp_bcast(n, ionode_id, intra_image_comm)
  if(n /= 432) stop 'Expected fixed 432-atom pair'
  allocate(x(3,n,2), g(3,n), z(n), u(3,n), displaced(3,n))
  if(ionode) then
    do i=1,3
      read(*,*) cell(:,i)
    end do
    do k=1,2
      do i=1,n
        read(*,*) z(i),x(:,i,k)
      end do
    end do
    do i=1,n
      read(*,*) u(:,i)
    end do
  end if
  call mp_bcast(cell, ionode_id, intra_image_comm)
  call mp_bcast(z, ionode_id, intra_image_comm)
  do k=1,2
    call mp_bcast(x(:,:,k), ionode_id, intra_image_comm)
  end do
  call mp_bcast(u, ionode_id, intra_image_comm)
  if(any(z /= 74)) stop 'Expected tungsten only'
  if(abs(sum(u*u)-1.0_wp) > 1.e-10_wp) stop 'Non-unit fixed check direction'
  x=x/abohr
  cell=cell/abohr
  inp%threebody=.true.
  inp%numgrad=.false.
  inp%cutoff=sqrt(9000.0_wp)
  inp%cutoff_cn=40.0_wp
  call dftd3_init(calc, inp)
  call dftd3_set_functional(calc, func='pbe', version=3, tz=.false.)
  if(ionode) then
    open(newunit=outunit, file='w_d3_raw.dat', status='new', action='write', iostat=status)
  end if
  call mp_bcast(status, ionode_id, intra_image_comm)
  if(status /= 0) stop 'Output exists or cannot be created'
  if(ionode) then
    write(outunit,'(A,2ES26.17)') 'units bohr_A hartree_eV ',abohr,AUTOEV
    write(outunit,'(A,5ES26.17)') 'parameters s6 rs6 s18 rs18 alp ', &
      calc%s6,calc%rs6,calc%s18,calc%rs18,calc%alp
    write(outunit,'(A,2ES26.17)') 'squared_cutoffs_bohr ',calc%rthr,calc%cn_thr
  end if
  do k=1,2
    stress=0.0_wp
    call dftd3_pbc_dispersion(calc,x(:,:,k),z,cell,e,g,stress)
    if(ionode) then
      write(outunit,'(A,I3,A,ES26.17)') 'geometry ',k-1,' energy_Ha ',e
      ! API returns positive energy gradients: force=-gradient in Ha/Bohr.
      ! QE forces.f90 multiplies by -2 to express Ry/Bohr.
      do i=1,n
        write(outunit,'(I6,3ES26.17)') i,-g(:,i)
      end do
      flush(outunit)
    end if
  end do
  projection=sum(g*u)  ! Gradient at fixed 50 fs along fixed structural displacement.
  do j=1,2
    h=1.0e-3_wp/(2.0_wp**(j-1))
    displaced=x(:,:,2)+h*u
    call dftd3_pbc_dispersion(calc,displaced,z,cell,ep)
    displaced=x(:,:,2)-h*u
    call dftd3_pbc_dispersion(calc,displaced,z,cell,em)
    if(ionode) write(outunit,'(A,4ES26.17)') 'gradient_check h analytic central difference ', &
      h,projection,(ep-em)/(2.0_wp*h),(ep-em)/(2.0_wp*h)-projection
  end do
  if(ionode) then
    write(outunit,'(A)') 'COMPLETE'
    close(outunit)
  end if
  call mp_global_end()
end program w_qe_d3_component
