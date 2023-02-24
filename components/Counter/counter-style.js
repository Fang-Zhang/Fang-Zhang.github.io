import { makeStyles } from 'tss-react/mui';

const counterStyles = makeStyles({ uniqId: 'counter' })(theme => ({
  counterWrap: {
    backgroundImage: `linear-gradient(120deg, ${theme.palette.mode === 'dark' ? theme.palette.secondary.dark : theme.palette.secondary.main}, ${theme.palette.mode === 'dark' ? theme.palette.primary.dark : theme.palette.primary.main})`,
    padding: theme.spacing(10, 0)
  },
  text: {
    display: 'flex',
    alignItems: 'center',
    [theme.breakpoints.down('sm')]: {
      justifyContent: 'center'
    }
  },
  counterItem: {
    color: theme.palette.common.white,
    [theme.breakpoints.up('sm')]: {
      paddingLeft: theme.spacing(5),
    },
    '& i': {
      marginRight: theme.spacing(2),
      color: theme.palette.common.white,
      fontSize: 40,
      lineHeight: '40px'
    }
  }
}));

// TODO jss-to-tss-react codemod: usages of this hook outside of this file will not be converted.
export default counterStyles;
