#include <Python.h>
#include "numpy/arrayobject.h"
#include "fieldize.h"

/*Check whether the passed array has type typename. Returns 1 if it doesn't, 0 if it does.*/
int check_type(PyArrayObject * arr, int npy_typename)
{
  return !PyArray_EquivTypes(PyArray_DESCR(arr), PyArray_DescrFromType(npy_typename));
}

//  int    3*nval arr  nval arr  nval arr   nx*nx arr  int     nval arr  (or 0)
//['nval','pos',     'radii',    'value',   'field',   'nx',    'weights']
PyObject * Py_SPH_Fieldize(PyObject *self, PyObject *args)
{
    PyArrayObject *pos, *radii, *value, *weights;
    int periodic, nx, ret;
    if(!PyArg_ParseTuple(args, "O!O!O!O!ii",&PyArray_Type, &pos, &PyArray_Type, &radii, &PyArray_Type, &value, &PyArray_Type, &weights,&periodic, &nx) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use pos, radii, value, weights periodic=False, nx\n");
        return NULL;
    }
    if(check_type(pos, NPY_FLOAT) || check_type(radii, NPY_FLOAT) || check_type(value, NPY_FLOAT) || check_type(weights, NPY_DOUBLE))
    {
          PyErr_SetString(PyExc_AttributeError, "Input arrays do not have appropriate type: pos, radii and value need float32, weights float64.\n");
          return NULL;
    }
    const npy_intp nval = PyArray_DIM(radii,0);
    if(nval != PyArray_DIM(value,0) || nval != PyArray_DIM(pos,0))
    {
      PyErr_SetString(PyExc_ValueError, "pos, radii and value should have the same length.\n");
      return NULL;
    }
//     int totlow=0, tothigh=0;
    //Field for the output.
    npy_intp size[2]={nx,nx};
    PyArrayObject * pyfield = (PyArrayObject *) PyArray_SimpleNew(2, size, NPY_DOUBLE);
    PyArray_FILLWBYTE(pyfield, 0);
    double * field = (double *) PyArray_DATA(pyfield);
    //Copy of field array to store compensated bits for Kahan summation
    double * comp = (double *) calloc(nx*nx,sizeof(double));
    if( !comp || !field ){
      PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for field arrays.\n");
      return NULL;
    }
    //Do the work
    ret = SPH_interpolate(field, comp, nx, pos, radii, value, weights, nval, periodic);
    free(comp);

    if( ret == 1 ){
      PyErr_SetString(PyExc_ValueError, "Massless particle detected!");
      return NULL;
    }
    //printf("Total high: %d total low: %d (%ld)\n",tothigh, totlow,nval);
    PyObject * for_return = Py_BuildValue("O",pyfield);
    Py_DECREF(pyfield);
    return for_return;
}


PyObject * Py_find_halo_kernel(PyObject *self, PyObject *args)
{
    PyArrayObject *sub_cofm, *sub_mass, *xcells, *ycells, *halo_mass;
    double celsz;
    if(!PyArg_ParseTuple(args, "O!O!O!O!O!d",&PyArray_Type, &sub_cofm, &PyArray_Type, &sub_mass, &PyArray_Type, &xcells, &PyArray_Type, &ycells,&PyArray_Type, &halo_mass, &celsz) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use sub_cofm, sub_mass, xcells, ycells, halo_mass, celsz\n");
        return NULL;
    }

    const npy_intp ncells = PyArray_DIM(xcells,0);
    for (int i=0; i< ncells; i++)
    {
        const int64_t xcoord =  (*(int64_t *) PyArray_GETPTR1(xcells,i));
        const int64_t ycoord =  (*(int64_t *) PyArray_GETPTR1(ycells,i));
        double dd_min = pow(*(double *) PyArray_GETPTR2(sub_cofm,0,1) - celsz*xcoord,2)
            + pow(*(double *) PyArray_GETPTR2(sub_cofm,0,2) - celsz*ycoord,2);
        int nearest_halo=0;
        for (int j=1; j < PyArray_DIM(sub_cofm,0); j++)
        {
            double dd = pow(*(double *) PyArray_GETPTR2(sub_cofm,j,1) - celsz*xcoord,2)
                        + pow(*(double *) PyArray_GETPTR2(sub_cofm,j,2) - celsz*ycoord,2);
            if (dd < dd_min){
                dd_min = dd;
                nearest_halo = j;
            }
        }
        *(double *) PyArray_GETPTR2(halo_mass,xcoord,ycoord) = *(double *) PyArray_GETPTR1(sub_mass,nearest_halo);
    }
    int i = 0;
    return Py_BuildValue("i",&i);
}

static PyMethodDef __fieldize[] = {
  {"_SPH_Fieldize", Py_SPH_Fieldize, METH_VARARGS,
   "Interpolate particles onto a grid using SPH interpolation."
   "    Arguments: pos, radii, value, weights, periodic=T/F, nx"
   "    "},
  {"_find_halo_kernel", Py_find_halo_kernel, METH_VARARGS,
   "Kernel for populating a field containing the mass of the nearest halo to each point"
   "    Arguments: sub_cofm, sub_mass, xcells, ycells (output from np.where), halo_mass[nn], celsz"
   "    "},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
init_fieldize_priv(void)
{
  Py_InitModule("_fieldize_priv", __fieldize);
  import_array();
}
