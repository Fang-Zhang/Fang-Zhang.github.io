import { makeStyles } from 'tss-react/mui';
import titleDecoLight from '/public/images/profile/titleDecoLight.png';
import titleDecoDark from '/public/images/profile/titleDecoDark.png';

const titleStyles = makeStyles({ uniqId: 'title' })((theme, _params, classes) => ({
  dark: {},
  title: {
    display: 'block',
    position: 'relative',
    marginBottom: theme.spacing(3),
    textAlign: 'center',
    height: theme.spacing(8),
    background: `url(${theme.palette.mode === 'dark' ? titleDecoDark : titleDecoLight}) no-repeat center center`,
    backgroundSize: 72,
    padding: theme.spacing(2, 0),
    '& h4': {
      color: theme.palette.text.primary,
      verticalAlign: 'middle',
      fontWeight: theme.typography.fontWeightRegular,
      textTransform: 'capitalize',
      fontSize: 36,
      lineHeight: '56px',
      [theme.breakpoints.down('lg')]: {
        fontSize: 32,
        lineHeight: '48px'
      },
      [theme.breakpoints.down('sm')]: {
        fontSize: 24,
        lineHeight: '36px',
      },
      '& strong': {
        fontWeight: theme.typography.fontWeightBold,
      }
    },
    [`&.${classes.dark}`]: {
      '& h4': {
        color: theme.palette.common.white,
      }
    }
  }
}));

// TODO jss-to-tss-react codemod: usages of this hook outside of this file will not be converted.
export default titleStyles;
